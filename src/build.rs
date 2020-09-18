use anyhow::{anyhow, ensure, Context};
use hporecord::{EvalRecord, ParamRange, Record, Scale, StudyRecord};
use kurobako_core::domain;
use kurobako_core::problem::{ProblemSpec, ProblemSpecBuilder};
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use randomforest::criterion::Mse;
use randomforest::table::{ColumnType, TableBuilder};
use randomforest::{RandomForestRegressor, RandomForestRegressorOptions};
use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
pub struct BuildOpt {
    /// Problem name (Lua script).
    #[structopt(long)]
    pub problem_name: String,

    /// Output directory.
    #[structopt(long, default_value = "result/")]
    pub out: PathBuf,

    /// Objective value index.
    #[structopt(long, default_value = "0")]
    pub objective_index: usize,

    /// Max samples used for building each tree in a random forest.
    #[structopt(long, default_value = "1000")]
    pub max_samples: NonZeroUsize,

    /// Number of trees in a rando forest.
    #[structopt(long, default_value = "1000")]
    pub trees: NonZeroUsize,

    #[structopt(long, default_value = "0.1")]
    pub test_rate: f64,

    #[structopt(long)]
    pub dump_csv: bool,
}

impl BuildOpt {
    pub fn build(&self, records: &[Record]) -> anyhow::Result<()> {
        let mut problem_names = BTreeMap::new();
        let mut problems = BTreeMap::new();
        for record in records {
            if let Record::Study(study) = record {
                let lua = rlua::Lua::new();
                let name: String = lua.context(|lua_ctx| {
                    let globals = lua_ctx.globals();

                    // TODO
                    globals.set("attrs", study.attrs.clone())?;

                    lua_ctx.load(&self.problem_name).eval()
                })?;
                let problem = problems
                    .entry(name.clone())
                    .or_insert_with(|| Problem::new(name, study, self));
                problem_names.insert(&study.id, problem.name.clone());
            }
        }

        for record in records {
            match record {
                Record::Study(_) => {}
                Record::Eval(eval) => {
                    if !eval.state.is_complete() {
                        continue;
                    }
                    let name = problem_names
                        .get(&eval.study)
                        .ok_or_else(|| anyhow!("unknown study {:?}", eval.study))?;
                    let problem = problems.get_mut(name).expect("unreachable");

                    let v = problem.get_value(eval)?;
                    if v.is_finite() {
                        problem.rows.push((eval.params.clone(), v));
                        problem.samples += 1;
                    }
                }
            }
        }

        let problems_len = problems.len();
        for (i, problem) in problems.values_mut().enumerate() {
            let dir = self.out.join(format!("{}/", problem.name));
            std::fs::create_dir_all(&dir)?;

            let rows = std::mem::take(&mut problem.rows);
            let model = problem.build_model(rows)?;

            let spec_path = dir.join("spec.json");
            let spec_file = std::fs::File::create(&spec_path)
                .with_context(|| format!("path={:?}", spec_path))?;
            serde_json::to_writer(spec_file, &model.spec)?;

            let regressor_path = dir.join("model.bin");
            let regressor_file = std::fs::File::create(&regressor_path)
                .with_context(|| format!("path={:?}", regressor_path))?;
            model.regressor.serialize(BufWriter::new(regressor_file))?;

            eprintln!(
                "[{}/{}] Created: {:?} (n={}, outliers={}, cc={})",
                i, problems_len, dir, model.samples, model.outliers, model.cc
            );
        }

        Ok(())
    }
}

#[derive(Debug)]
struct Problem<'a> {
    name: String,
    study: &'a StudyRecord,
    table: TableBuilder,
    rows: Vec<(Vec<f64>, f64)>,
    samples: usize,
    opt: &'a BuildOpt,
}

impl<'a> Problem<'a> {
    fn new(name: String, study: &'a StudyRecord, opt: &'a BuildOpt) -> Self {
        let mut table = TableBuilder::new();
        let column_types = study
            .params
            .iter()
            .map(|p| {
                if matches!(p.range, ParamRange::Categorical{..}) {
                    ColumnType::Categorical
                } else {
                    ColumnType::Numerical
                }
            })
            .collect::<Vec<_>>();
        table
            .set_feature_column_types(&column_types)
            .expect("unreachable");
        Self {
            name,
            study,
            table,
            samples: 0,
            rows: Vec::new(),
            opt,
        }
    }

    fn get_value(&self, record: &EvalRecord) -> anyhow::Result<f64> {
        ensure!(
            self.opt.objective_index < record.values.len(),
            "too large objective index"
        );

        let mut v = record.values[self.opt.objective_index];
        if self.study.values[self.opt.objective_index]
            .direction
            .is_maximize()
        {
            v = -v;
        }
        Ok(v)
    }

    fn build_model(&mut self, mut rows: Vec<(Vec<f64>, f64)>) -> anyhow::Result<Model> {
        let value_def = &self.study.values[self.opt.objective_index];
        let params = self
            .study
            .params
            .iter()
            .map(|p| {
                let mut v = domain::var(&p.name);
                match &p.range {
                    ParamRange::Categorical { choices } => v.categorical(choices.iter()),
                    ParamRange::Numerical {
                        min,
                        max,
                        step,
                        scale,
                    } => {
                        if let Some(step) = step {
                            assert_eq!(*step, 1.0, "not implemented");
                            v = v.discrete(*min as i64, *max as i64 + 1);
                        } else {
                            v = v.continuous(*min, *max);
                        }
                        if *scale == Scale::Log {
                            v = v.log_uniform();
                        }
                        v
                    }
                }
            })
            .collect::<Vec<_>>();

        let rows_len = rows.len();
        rows.shuffle(&mut rand::thread_rng());
        let (test_rows, train_rows) =
            rows.split_at_mut((rows_len as f64 * self.opt.test_rate) as usize);

        let mut outliers = 0;
        let p95 = percentile(train_rows.iter().map(|x| x.1), 0.95);
        for r in train_rows.iter_mut() {
            if r.1 > p95 {
                r.1 = p95;
                outliers += 1;
            }
        }

        for r in train_rows {
            self.table.add_row(&r.0, r.1)?;
        }

        let table = self.table.build()?;

        let regressor = RandomForestRegressorOptions::new()
            .parallel()
            .max_samples(self.opt.max_samples)
            .trees(self.opt.trees)
            .fit(Mse, table);

        let cc = self.spearman_rank_correlation_coefficient(&regressor, &test_rows);

        let spec = ProblemSpecBuilder::new(&self.name)
            .params(params)
            .attr("samples", &self.samples.to_string())
            .attr("outliers", &outliers.to_string())
            .attr("Spearman's rank correlation coefficient", &cc.to_string())
            .value(
                domain::var(&value_def.name).continuous(value_def.range.min, value_def.range.max),
            )
            .finish()?;

        Ok(Model {
            spec,
            regressor,
            samples: self.samples,
            outliers,
            cc,
        })
    }

    fn spearman_rank_correlation_coefficient(
        &self,
        regressor: &RandomForestRegressor,
        test_rows: &[(Vec<f64>, f64)],
    ) -> f64 {
        let mut pairs = test_rows
            .iter()
            .map(|row| (0, row.1, regressor.predict(&row.0)))
            .collect::<Vec<_>>();

        if self.opt.dump_csv {
            let dir = self.opt.out.join(format!("{}/", self.name));

            let csv_path = dir.join("predict.csv");
            let csv_file = std::fs::File::create(&csv_path)
                .with_context(|| format!("path={:?}", csv_path))
                .expect("TODO");
            let mut csv_file = BufWriter::new(csv_file);

            writeln!(csv_file, "True Value,Predicted Value").expect("TODO");
            for (_, x, y) in &pairs {
                writeln!(csv_file, "{},{}", x, y).expect("TODO");
            }
        }

        pairs.sort_by_key(|x| OrderedFloat(x.1));
        for (i, (true_rank, _, _)) in pairs.iter_mut().enumerate() {
            *true_rank = i;
        }

        pairs.sort_by_key(|x| OrderedFloat(x.2));
        let a = pairs
            .iter()
            .enumerate()
            .map(|(expected_rank, (true_rank, _, _))| {
                (*true_rank as f64 - expected_rank as f64).abs().powi(2)
            })
            .sum::<f64>();
        let n = pairs.len() as f64;

        1.0 - (6.0 * a) / (n.powi(3) - n)
    }
}

#[derive(Debug)]
struct Model {
    spec: ProblemSpec,
    regressor: RandomForestRegressor,
    samples: usize,
    outliers: usize,
    cc: f64,
}

fn percentile(xs: impl Iterator<Item = f64>, p: f64) -> f64 {
    let mut xs = xs.collect::<Vec<_>>();
    xs.sort_by_key(|x| OrderedFloat(*x));
    xs[(xs.len() as f64 * p) as usize]
}
