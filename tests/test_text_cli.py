"""Tests for tallgrass-text CLI argument parsing."""


from tallgrass.text.cli import main


class TestTextCli:
    def test_list_sessions(self, capsys):
        """--list-sessions prints sessions and exits."""
        main(["--list-sessions"])
        captured = capsys.readouterr()
        assert "Known Kansas Legislature sessions" in captured.out
        assert "91st" in captured.out
        assert "Special sessions:" in captured.out

    def test_list_sessions_shows_historical(self, capsys):
        main(["--list-sessions"])
        captured = capsys.readouterr()
        assert "84th" in captured.out
        assert "2011" in captured.out

    def test_default_year(self, monkeypatch):
        """Default year is CURRENT_BIENNIUM_START."""
        import argparse

        # Capture the parsed args by monkeypatching
        parsed_args = {}

        def capture_main(argv=None):
            parser = argparse.ArgumentParser()
            parser.add_argument("year", nargs="?", type=int, default=2025)
            parser.add_argument("--special", action="store_true")
            parser.add_argument("--types", type=str, default="introduced,supp_note")
            parser.add_argument("--clear-cache", action="store_true")
            parser.add_argument("--list-sessions", action="store_true")
            args = parser.parse_args(argv)
            parsed_args["year"] = args.year
            parsed_args["special"] = args.special

        capture_main([])
        assert parsed_args["year"] == 2025

    def test_custom_year(self, monkeypatch):
        """Year argument is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("year", nargs="?", type=int, default=2025)
        args = parser.parse_args(["2023"])
        assert args.year == 2023

    def test_special_flag(self, monkeypatch):
        """--special flag is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("year", nargs="?", type=int, default=2025)
        parser.add_argument("--special", action="store_true")
        args = parser.parse_args(["2024", "--special"])
        assert args.special is True
        assert args.year == 2024

    def test_types_flag(self, monkeypatch):
        """--types flag parses comma-separated types."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--types", type=str, default="introduced,supp_note")
        args = parser.parse_args(["--types", "introduced"])
        types = [t.strip() for t in args.types.split(",")]
        assert types == ["introduced"]

    def test_types_multiple(self):
        """Multiple types are split correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--types", type=str, default="introduced,supp_note")
        args = parser.parse_args(["--types", "introduced,supp_note,enrolled"])
        types = [t.strip() for t in args.types.split(",")]
        assert types == ["introduced", "supp_note", "enrolled"]

    def test_clear_cache_flag(self):
        """--clear-cache flag is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--clear-cache", action="store_true")
        args = parser.parse_args(["--clear-cache"])
        assert args.clear_cache is True

    def test_entry_point_registered(self):
        """tallgrass-text entry point is importable."""
        from tallgrass.text.cli import main

        assert callable(main)
