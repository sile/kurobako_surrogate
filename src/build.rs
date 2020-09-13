use anyhow::{anyhow, ensure, Context};
use hporecord::{EvalRecord, ParamRange, Record, Scale, StudyRecord};
use kurobako_core::domain;
use kurobako_core::problem::{ProblemSpec, ProblemSpecBuilder};
use randomforest::criterion::Mse;
use randomforest::table::{ColumnType, TableBuilder};
use randomforest::{RandomForestRegressor, RandomForestRegressorOptions};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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
                    .or_insert_with(|| Problem::new(name, study, self.objective_index));
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
                        problem.table.add_row(&eval.params, v)?;
                        problem.samples += 1;
                    }
                }
            }
        }

        std::fs::create_dir_all(&self.out)?;
        for (i, problem) in problems.values().enumerate() {
            let path = self.out.join(format!("{}.json", problem.name));

            let model = problem.build_model()?;
            let file = std::fs::File::create(&path).with_context(|| format!("path={:?}", path))?;
            serde_json::to_writer(file, &model)?;

            eprintln!("[{}/{}] Created: {:?}", i, problems.len(), path);
        }

        Ok(())
    }
}

#[derive(Debug)]
struct Problem<'a> {
    name: String,
    study: &'a StudyRecord,
    objective_index: usize,
    table: TableBuilder,
    samples: usize,
}

impl<'a> Problem<'a> {
    fn new(name: String, study: &'a StudyRecord, objective_index: usize) -> Self {
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
            objective_index,
            table,
            samples: 0,
        }
    }

    fn get_value(&self, record: &EvalRecord) -> anyhow::Result<f64> {
        ensure!(
            self.objective_index < record.values.len(),
            "too large objective index"
        );

        let mut v = record.values[self.objective_index];
        if self.study.values[self.objective_index]
            .direction
            .is_maximize()
        {
            v = -v;
        }
        Ok(v)
    }

    fn build_model(&self) -> anyhow::Result<Model> {
        let value_def = &self.study.values[self.objective_index];
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

        let spec = ProblemSpecBuilder::new(&self.name)
            .params(params)
            .attr("samples", &self.samples.to_string())
            .value(
                domain::var(&value_def.name).continuous(value_def.range.min, value_def.range.max),
            )
            .finish()?;

        // TODO: CV
        eprintln!("SAMPLES: {}", self.samples);
        let regressor = RandomForestRegressorOptions::new()
            .parallel()
            .max_samples(NonZeroUsize::new(1000).expect("unreachable")) // TODO: option
            .trees(NonZeroUsize::new(1000).expect("unreachable")) // TODO: option
            .fit(Mse, self.table.build()?);

        Ok(Model { spec, regressor })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Model {
    pub spec: ProblemSpec,
    pub regressor: RandomForestRegressor,
}
