from argparse import ArgumentParser
from .model import ResMem, transformer
from .utils import DirectoryImageset
from torch.utils.data import DataLoader
import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.argument('layer', type=click.INT)
@click.option('--output', type=click.Path(), help='Output csv')
def resmem_features(directory, layer, output=None):
    """Get resmem activations for all filters in a layer, for all images in a directory."""
    import pandas as pd
    dataset = DirectoryImageset(directory, transform=transformer)
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    model = ResMem(pretrained=True)
    model.eval()
    outputs = {}
    for name, image in dl:
        activations = model.resnet_activation(image.view(-1, 3, 227, 227), layer).detach().cpu().numpy().flatten()
        outputs[name] = activations
    df = pd.DataFrame.from_dict(outputs, orient='index')
    if output:
        df.to_csv(output)
    else:
        df.to_csv(f'{directory}_resmem_features.csv')


cli.add_command(resmem_features)
if __name__ == "__main__":
    cli()
