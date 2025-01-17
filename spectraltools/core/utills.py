from torch.nn import Linear, Sequential, Conv2d
from .spectraldense import Spectral

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

def spectral_linear(model,old_layer=Linear, new_layer=Spectral, verbose=True):
  for name, module in model.named_modules():
          if isinstance(module, old_layer):
            # Get current bn layer
            ol = get_layer(model, name)
            if ol.out_features == 13:
              print("here")
            else:
              # Crate new gn layer
              nl = Sequential(ol, new_layer(ol.out_features))
              # Assign gn
              if verbose:
                print("Swapping {} with {}".format(ol, nl))
              set_layer(model, name, nl)


def spectral_conv2d(model, old_layer=Conv2d, new_layer=Spectral, verbose=False):
  for name, module in model.named_modules():
    if isinstance(module, old_layer):
        # Get current bn layer
        ol = get_layer(model, name)
        # Create new gn layer
        nl = Sequential(ol, new_layer(ol.out_channels))
        # Assign gn
        if verbose:
          print("Swapping {} with {}".format(ol, nl))
        set_layer(model, name, nl)

def spectral_all(model, verbose=False):
  spectral_linear(model,verbose=verbose)
  spectral_conv2d(model,verbose=verbose)
