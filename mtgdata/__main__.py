#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2025 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

if __name__ == "__main__":
    import argparse
    import logging
    from mtgdata.scryfall import _make_parser_scryfall_prepare, _run_scryfall_prepare
    from mtgdata.scryfall_convert import (
        _make_parser_scryfall_convert,
        _run_scryfall_convert,
    )

    # initialise logging
    logging.basicConfig(level=logging.INFO)
    YLW = "\033[93m"
    RST = "\033[0m"

    # subcommands
    cli = argparse.ArgumentParser()
    parsers = cli.add_subparsers()
    parsers.required = True

    # subcommand: prepare -- add args from scryfall.py
    parser_prepare = parsers.add_parser("prepare")
    parser_prepare.set_defaults(
        _run_fn_=_run_scryfall_prepare, _run_msg_=f"{YLW}preparing...{RST}"
    )
    _make_parser_scryfall_prepare(parser_prepare)

    # subcommand: convert -- add args from scryfall_convert.py
    parser_convert = parsers.add_parser("convert")
    parser_convert.set_defaults(
        _run_fn_=_run_scryfall_convert, _run_msg_=f"{YLW}converting...{RST}"
    )
    _make_parser_scryfall_convert(parser_convert)

    # run the specified subcommand!
    args = cli.parse_args()
    print(f"{args._run_msg_}")
    args._run_fn_(args)
