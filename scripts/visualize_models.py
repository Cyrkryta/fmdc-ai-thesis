from torchview import draw_graph

from project.models.unet3d_direct import UNet3DDirect
from project.models.unet3d_fieldmap import UNet3DFieldmap


def _plot_model_graph(model, name):
    model_graph = draw_graph(model, input_size=(1, 2, 36, 64, 64), device='meta')
    graph = model_graph.visual_graph

    graph_pdf = graph.pipe(format='pdf')
    with open(f'/home/mlc/dev/fmdc/downloads/model-graph-{name}.pdf', 'wb') as f:
        f.write(graph_pdf)


if __name__ == '__main__':
    """
    Visualize all model's architectures as graphs.
    """

    _plot_model_graph(UNet3DDirect(), 'direct')
    _plot_model_graph(UNet3DFieldmap(), 'fieldmap')
