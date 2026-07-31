from cli import main


def test_greet(capsys):
    assert main(["cli.py", "--greet"]) == 0
    assert capsys.readouterr().out.strip() == "hello"
