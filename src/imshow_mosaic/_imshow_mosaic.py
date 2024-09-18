from typing import Any, Literal, Mapping, Optional, Union
import pandas as pd
import numpy as np
import rectpack
import matplotlib.pyplot as plt
import matplotlib.patches
import skimage.util.dtype
import skimage.measure


def _clean_data(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    result = pd.DataFrame(None, index=data.index)
    for k, v in kwargs.items():
        if isinstance(k, str):
            try:
                # First try to treat v as a key in data
                result[k] = data[v]
                continue
            except KeyError:
                pass

        # otherwise, use the value itself

        if (
            not isinstance(v, str)
            and isinstance(v, (list, tuple, np.ndarray))
            and len(v) != len(data)
        ):
            v = [v] * len(data)

        result[k] = v

    return result


def _apply_defaults(data: pd.DataFrame, **kwargs):
    for k, v in kwargs.items():
        if k not in data:
            data[k] = v


def _init_wh(data: pd.DataFrame):
    if "w" not in data:
        data["w"] = data["img"].apply(lambda img: img.shape[1])
    if "h" not in data:
        data["h"] = data["img"].apply(lambda img: img.shape[0])


def imshow_mosaic(
    img,
    *,
    cx=None,
    cy=None,
    w=None,
    h=None,
    label=None,
    debug=None,
    data=None,
    shuffle: bool = True,
    grid_size=50,
    width: int = 10,
    height: int = 10,
    zoom: float = 1.0,
    scalebar_len_px: Optional[int] = None,
    scalebar_text: Optional[str] = None,
    label_kwargs=None,
    packrect_color=None,
    packrect_kwargs: Optional[Mapping[str, Any]] = None,
    draw_obj_boundaries=False,
    draw_centroid=False,
    draw_img_boundaries=False,
    clip_packrect=False,
    clip_ax=True,
    bgcolor=None,
    ax_frame_on=False,
    ax: Optional[plt.Axes] = None,  # type: ignore
    verbose=False,
    random_state=None,
    background_color=None,
):
    """
    Plot images in a space-saving mosaic.

    Args:
        img (vector or key in ``data``): Images that will be shown
        cx, cy (vector or key in ``data``): Relative center of gravity (between 0 and 1).
        w, h (vector or key in ``data``): Width and height of an object
            centered around the center of gravity.
            (Can be different than the actual image dimensions.)
        label (vector or key in ``data``): Label for each object.
        debug
        data (pandas.DataFrame, numpy.ndarray, mapping, or sequence):
            Input data structure as a collection of vectors that can be assigned to named variables.
        shuffle (bool): Shuffle the data before packing.
        grid_size (int): Granularity of the grid.
            Each object receives a minimum area of grid_size x grid_size.
        width, height (int): Dimensions of the drawing surface in grid cells.
        zoom (float): Zoom factor for the objects.
        label_kwargs (dict, optional): Kwargs passed to ax.annotate.
        draw_packrect (bool): Draw the rectangle that was reserved for an object.
        packrect_kwargs (None, optional): Additional keywords for matplotlib.patches.Rectangle for the pacrect.
        draw_obj_boundaries (bool): Draw the area that is occupied by an object (according to ``w`` and ``h``).
        draw_centroid (bool): Draw the centroid of an object.
        draw_img_boundaries (bool): Draw the outer border of an image.
            (Can be different than the supplied object dimensions.)
        clip_packrect (bool): Clip images to their packrect.
        clip_ax (bool): Clip elements to the axis.
        bgcolor: Background color for the axis.
        ax_frame_on: Show axis frame (default: False).
        ax (matplotlib.axes.Axes): Pre-existing axes for the plot.
            Otherwise, call matplotlib.pyplot.gca() internally.
        verbose (bool): Be verbose.
        random_state: Set for reproducibility.
    """

    if ax is None:
        ax = plt.gca()

    if label_kwargs is None:
        label_kwargs = {}

    if packrect_kwargs is None:
        packrect_kwargs = {}

    data = _clean_data(
        data,
        img=img,
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        label=label,
        debug=debug,
        packrect_color=packrect_color,
        background_color=background_color,
    )
    _apply_defaults(data, cx=0.5, cy=0.5, label=None)
    _init_wh(data)

    if shuffle:
        data = data.sample(frac=1, random_state=random_state)

    # Calculate size in terms of grid_size
    data[["w_grid", "h_grid"]] = data[["w", "h"]] / grid_size
    # ...and round to whole grid cells for packing
    data[["w_grid_i", "h_grid_i"]] = (
        data[["w_grid", "h_grid"]].round().clip(1).astype(int)
    )

    # Setup bin packing
    packer = rectpack.newPacker(rotation=False, sort_algo=rectpack.SORT_NONE)
    for row in data.itertuples():
        packer.add_rect(row.w_grid_i, row.h_grid_i, row.Index)

    packer.add_bin(width, height)
    packer.pack()  # type: ignore

    packed = pd.DataFrame(
        packer.rect_list(), columns=["bin", "x_grid", "y_grid", "h", "w", "index"]
    )

    # Merge pack info back into data
    data = data.merge(
        packed[["x_grid", "y_grid", "index"]], left_index=True, right_on="index"
    )

    for obj in data.itertuples():
        # Calculate centroid (in data coordinates)
        _cx = obj.x_grid + obj.w_grid_i / 2
        _cy = obj.y_grid + obj.h_grid_i / 2

        if verbose:
            print(f"Object {obj.Index} at ({_cx}, {_cy})")

        # Calculate extent of the image (in data coordinates)
        _w = obj.img.shape[1] / grid_size * zoom
        _h = obj.img.shape[0] / grid_size * zoom
        _x = _cx - _w * (obj.cx)
        _y = _cy - _h * (1 - obj.cy)

        im_artist = ax.imshow(
            obj.img,
            extent=[_x, _x + _w, _y, _y + _h],  # transform=ax.transData
            clip_on=clip_ax,
            # Put images on top
            zorder=1,
        )

        if draw_img_boundaries:
            r = matplotlib.patches.Rectangle(
                (_x, _y),
                _w,
                _h,
                ec="green",
                fc="None",
                clip_on=clip_ax,
            )
            ax.add_artist(r)

        if draw_centroid:
            ax.plot(_cx, _cy, "rx")

        if obj.label is not None:
            verticalalignment = label_kwargs.pop("verticalalignment", "center")
            ax.annotate(
                obj.label,
                (_cx, obj.y_grid if verticalalignment == "top" else _cy),
                ha="center",
                verticalalignment=verticalalignment,
                clip_on=clip_ax,
                **label_kwargs,
            )

            if verbose:
                print(f"  {obj.label}")

        if obj.background_color is not None:
            background = matplotlib.patches.Rectangle(
                (obj.x_grid, obj.y_grid),
                obj.w_grid_i,
                obj.h_grid_i,
                fc=obj.background_color,
                transform=ax.transData,
                clip_on=True,
                zorder=0,
            )
            ax.add_artist(background)

        # packrect for image clipping and drawing
        obj_packrect_kwargs = {**packrect_kwargs, "edgecolor": obj.packrect_color}
        packrect = matplotlib.patches.Rectangle(
            (obj.x_grid, obj.y_grid),
            obj.w_grid_i,
            obj.h_grid_i,
            fc="None",
            transform=ax.transData,
            # clip_on=False,
            **obj_packrect_kwargs,
        )

        if clip_packrect:
            im_artist.set_clip_path(packrect)

        if obj.packrect_color is not None:
            ax.add_artist(packrect)
            packrect.set_clip_path(packrect)

        if draw_obj_boundaries:
            r = matplotlib.patches.Rectangle(
                (_cx - obj.w_grid / 2, _cy - obj.h_grid / 2),
                obj.w_grid,
                obj.h_grid,
                ec="r",
                fc="None",
                clip_on=clip_ax,
            )
            ax.add_artist(r)

    if scalebar_len_px is not None:
        ylim = -0.25
        r = matplotlib.patches.Rectangle(
            (0, -0.25),
            scalebar_len_px / grid_size,
            0.25,
            ec="k",
            fc="k",
            # clip_on=clip_ax,
        )
        ax.add_artist(r)

        if scalebar_text is not None:
            ax.annotate(
                scalebar_text,
                (0, -0.125),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                color="white",
            )
    else:
        ylim = 0

    ax.set_aspect("equal")

    if bgcolor is not None:
        ax.set_facecolor(bgcolor)

    if not ax_frame_on:
        # ax.axis("off") does not work as it also removes the facecolor
        plt.setp(ax.spines.values(), visible=False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Set correct limits
    # ax.set_xlim(0, (data["x_grid"] + data["w_grid_i"]).max())
    # ax.set_ylim(ylim, (data["y_grid"] + data["h_grid_i"]).max())
    ax.set_xlim(0, width)
    ax.set_ylim(height, ylim)


def background(img: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate the static background color of an image.

    Args:
        img (np.ndarray): Image of shape (h,w) or (h,w,c).

    Returns:
        background (float or np.ndarray): Static background color.
            Scalar or vector of shape (c,).
    """

    if img.ndim == 2:
        return np.median(img)
    if img.ndim == 3:
        return np.median(img, (0, 1))

    raise ValueError(f"Unexpected shape: {img.shape}")


def image2alpha(
    img: np.ndarray,
    bg: Union[None, np.ndarray, float] = None,
    foreground: Literal[None, "light", "dark"] = None,
):
    """
    Convert a grayscale image with uniform background into an RGBA image and a solid background color.

    Args:
        img (np.ndarray): Image of shape (h,w) or (h,w,c).
        bg (optional): Background color.
    """

    if bg is None:
        bg = background(img)

    halftone = skimage.util.dtype._convert(img >= bg, dtype=img.dtype)
    rgba = np.dstack([halftone] * 4)
    alpha = skimage.util.dtype._convert(
        np.abs((img - bg) / (halftone - bg)), dtype=img.dtype
    )

    if foreground == "light":
        # Hide dark parts
        alpha[img < bg] = 0
    elif foreground == "dark":
        # Hide light parts
        alpha[img >= bg] = 0

    rgba[:, :, -1] = alpha

    return rgba, bg


def objectness(img: np.ndarray, bg: Union[None, np.ndarray, float] = None):
    """
    Calculate the objectness for an image containing a single object.

    Args:
        img (np.ndarray): Image of shape (h,w).
        bg (optional): Background color.
    """

    if bg is None:
        bg = background(img)

    img = np.abs(img - bg)

    if img.ndim == 2:
        return img

    return img.mean(axis=-1)


def centroid_dimensions(
    img: np.ndarray, padding=0, bg: Union[None, np.ndarray, float] = None
):
    """
    Calculate centroid and dimensions for an image containing a single object.

    The object is assumed to differ from the background color.

    Args:
        img (np.ndarray): Image of shape (h,w).
        padding (int, optional): Padding around an object (in px).
        bg (optional): Background color.

    Returns:
        Tuple (cx, cy, w, h)
    """

    img = objectness(img, bg)

    M = skimage.measure.moments(img, 2)
    cy, cx = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

    it = skimage.measure.inertia_tensor(img)

    lmb, vec = np.linalg.eigh(it)
    dimensions = np.abs(vec.dot(np.diag(2 * np.sqrt(lmb))))
    w, h = 2 * (dimensions.max(axis=1) + padding)

    w = min(w, img.shape[1])
    h = min(h, img.shape[0])

    return cx / img.shape[1], cy / img.shape[0], w, h
