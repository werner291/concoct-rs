mod cli;

use clap::Parser;
use cli::Args;

fn main() {
    let args = Args::parse();

    if let Err(e) = args.validate() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }

    // TODO: wire up pipeline stages
    eprintln!("concoct-rs: parsed args successfully");
    eprintln!("{args:#?}");
}
