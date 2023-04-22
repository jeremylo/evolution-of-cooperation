import click


@click.command()
def main():
    """This is a command-line tool for running experiments related to
    Jeremy Lo Ying Ping's final-year MEng project work at UCL on the subject
    of exploring the evolution of coooperative structures and social circles
    in societies of artificial learning-based agents."""

    click.secho("Hello, world!", fg="cyan", bold=True)


if __name__ == "__main__":
    main()
