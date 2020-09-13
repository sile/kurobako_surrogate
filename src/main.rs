use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
enum Opt {
    Build(kurobako_surrogate::build::BuildOpt),
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    match opt {
        Opt::Build(opt) => {
            let records = hporecord::io::read_records(std::io::stdin().lock())
                .collect::<anyhow::Result<Vec<_>>>()?;
            opt.build(&records)?;
        }
    }
    Ok(())
}
