"""Tests for CLI commands."""

from click.testing import CliRunner

from photo_indexer.cli import main


def test_cli_help():
    """Test CLI help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Photo Indexer" in result.output
    assert "index" in result.output
    assert "analyze" in result.output
    assert "person" in result.output
    assert "match-faces" in result.output
    assert "person-faces" in result.output


def test_person_help():
    """Test person command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["person", "--help"])
    assert result.exit_code == 0
    assert "Manage known persons for facial recognition" in result.output
    assert "--create" in result.output
    assert "--list" in result.output
    assert "--get" in result.output
    assert "--embedding-file" in result.output


def test_person_list_help():
    """Test person list command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["person", "list", "--help"])
    assert result.exit_code == 0
    # list is not a subcommand, it's an option, so this will show person help
    assert "Manage known persons" in result.output


def test_match_faces_help():
    """Test match-faces command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["match-faces", "--help"])
    assert result.exit_code == 0
    assert "Match detected faces to known persons" in result.output
    assert "--asset-id" in result.output
    assert "--all" in result.output
    assert "--threshold" in result.output


def test_person_faces_help():
    """Test person-faces command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["person-faces", "--help"])
    assert result.exit_code == 0
    assert "Show all faces matched to a specific person" in result.output
    assert "--name" in result.output


def test_main_metrics_port_option():
    """Test that --metrics-port option is accepted."""
    runner = CliRunner()
    # Just verify the option is recognized
    result = runner.invoke(main, ["--metrics-port", "9090", "--help"])
    assert result.exit_code == 0


def test_tag_help():
    """Test tag command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["tag", "--help"])
    assert result.exit_code == 0
    assert "Add, remove, or list tags for an asset" in result.output


def test_tag_import_help():
    """Test tag-import command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["tag-import", "--help"])
    assert result.exit_code == 0
    assert "Bulk import tags from a file" in result.output


def test_tags_help():
    """Test tags command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["tags", "--help"])
    assert result.exit_code == 0
    assert "Find assets by tag" in result.output