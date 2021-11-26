

if __name__ == '__main__':
    import argparse
    import logging
    from mtgdata.scryfall import _make_parser_scryfall_prepare, _run_scryfall_prepare
    from mtgdata.scryfall_convert import _make_parser_scryfall_convert, _run_scryfall_convert

    # initialise logging
    logging.basicConfig(level=logging.INFO)
    YLW = '\033[93m'
    RST = '\033[0m'

    # subcommands
    cli = argparse.ArgumentParser()
    parsers = cli.add_subparsers()
    parsers.required = True

    # subcommand: prepare -- add args from scryfall.py
    parser_prepare = parsers.add_parser('prepare')
    parser_prepare.set_defaults(_run_fn_=_run_scryfall_prepare, _run_msg_=f'{YLW}preparing...{RST}')
    _make_parser_scryfall_prepare(parser_prepare)

    # subcommand: convert -- add args from scryfall_convert.py
    parser_convert = parsers.add_parser('convert')
    parser_convert.set_defaults(_run_fn_=_run_scryfall_convert, _run_msg_=f'{YLW}converting...{RST}')
    _make_parser_scryfall_convert(parser_convert)

    # run the specified subcommand!
    args = cli.parse_args()
    print(f'{args._run_msg_}')
    args._run_fn_(args)
