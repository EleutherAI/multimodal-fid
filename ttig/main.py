from ttig.mmfid import make_reference_statistics
from ttig.model import MultiModalFeatureExtractor
import typer

app = typer.Typer()


@app.command()
def mmfid():
    pass


@app.command()
def calc_stats(name: str, folder_fp: str, num_samples: int = 500_000, batch_size: int = 128):
    model = MultiModalFeatureExtractor()
    return make_reference_statistics(
        name,
        model,
        folder_fp,
        num_samples,
        batch_size
    )


if __name__ == '__main__':
    app()
