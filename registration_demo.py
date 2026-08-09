"""Teaching helpers for the registration lecture.

Four things:

    deform(image, kind)          apply a known deformation, return the ground truth
    register(fixed, moving, kind) recover a transform between two images
    recover(image, kind)          deform then register, report the residual error in pixels
    animate(history, fixed)       show the optimiser walking downhill, iteration by iteration

`kind` is one of "rigid", "similarity", "affine", "nonlinear" throughout.

Direction convention: SimpleITK transforms are *pull* transforms, mapping output
coordinates back to input coordinates, so resampling with `T` moves the picture
by `T` inverse. `deform` hides this — its angle/translation/scale arguments
describe how the picture visibly moves — but the `.transform` it returns is the
pull transform that was handed to Resample. `register` returns the same flavour,
so the two compose directly in `transform_error`.

Run this file to regenerate the demo GIFs.
"""

from collections import namedtuple

import numpy as np
import SimpleITK as sitk

Deformation = namedtuple("Deformation", "image transform field")
Registration = namedtuple("Registration", "image transform history stop")
Step = namedtuple("Step", "iteration metric image")
Recovery = namedtuple("Recovery", "image error deformed truth registration")

KINDS = ("rigid", "similarity", "affine", "nonlinear")

# Deliberately different per kind, so the four demos don't look like the same
# picture nudged four times — each one shows off what its transform can do.
DEFAULTS = {
    "rigid": dict(angle=25.0, translation=(30.0, -22.0)),
    "similarity": dict(angle=-18.0, translation=(-24.0, 16.0), scale=1.30),
    "affine": dict(angle=15.0, translation=(20.0, -26.0), scale=0.85, shear=-0.35),
    # Coarser mesh than you might expect: a 4x4 mesh at high amplitude gives big,
    # sweeping warps that are still recoverable, where a finer mesh at the same
    # amplitude just traps the optimiser in a local minimum.
    "nonlinear": dict(mesh=(4, 4), amplitude=28.0, seed=0),
}


def _as_image(image):
    if isinstance(image, sitk.Image):
        return image
    return sitk.GetImageFromArray(np.asarray(image, dtype=np.float32))


def _centre(img):
    return img.TransformContinuousIndexToPhysicalPoint(
        [s / 2.0 for s in img.GetSize()]
    )


def _resample(moving, reference, transform):
    return sitk.GetArrayFromImage(
        sitk.Resample(moving, reference, transform, sitk.sitkLinear, 0.0)
    )


def deform(image, kind="rigid", **params):
    """Apply a known deformation of `kind` to a 2D image.

    Returns (image, transform, field): the deformed array, the ground-truth
    transform, and its dense displacement field, shaped (H, W, 2) in (x, y).

    Keyword arguments override `DEFAULTS[kind]`: angle (degrees), translation
    (x, y in pixels), scale, shear for the linear kinds; mesh, amplitude, seed
    for "nonlinear".
    """
    img = _as_image(image)
    p = {**DEFAULTS[kind], **params}

    if kind == "nonlinear":
        # Random B-spline control-point offsets: smooth, local, no clean inverse.
        transform = sitk.BSplineTransformInitializer(img, list(p["mesh"]))
        rng = np.random.default_rng(p["seed"])
        transform.SetParameters(
            rng.normal(0.0, p["amplitude"], len(transform.GetParameters()))
        )
    else:
        centre = _centre(img)
        angle = np.deg2rad(p["angle"])
        if kind == "rigid":
            visible = sitk.Euler2DTransform(centre, angle, p["translation"])
        elif kind == "similarity":
            visible = sitk.Similarity2DTransform()
            visible.SetCenter(centre)
            visible.SetAngle(angle)
            visible.SetScale(p["scale"])
            visible.SetTranslation(p["translation"])
        elif kind == "affine":
            rot = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            skew = np.array([[p["scale"], p["shear"]], [0.0, p["scale"]]])
            visible = sitk.AffineTransform(2)
            visible.SetCenter(centre)
            visible.SetMatrix((rot @ skew).ravel())
            visible.SetTranslation(p["translation"])
        else:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        # Invert so the arguments describe how the picture visibly moves.
        transform = visible.GetInverse()

    field = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        img.GetSize(),
        img.GetOrigin(),
        img.GetSpacing(),
        img.GetDirection(),
    )
    return Deformation(
        _resample(img, img, transform), transform, sitk.GetArrayFromImage(field)
    )


def register(fixed, moving, kind="rigid", capture=False, iterations=500, mesh=None):
    """Register `moving` onto `fixed` with a transform of `kind`.

    Returns (image, transform, history, stop): the resampled moving image, the
    recovered transform, one Step per optimiser iteration when `capture` is
    True (for `animate`), and why the optimiser stopped. Always read `stop` —
    "Maximum number of iterations" means it gave up rather than converged, and
    the transform will be wrong without looking wrong. Capture resamples the
    moving image every iteration, so leave it off when you only want numbers.
    """
    f, m = _as_image(fixed), _as_image(moving)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMeanSquares()
    method.SetInterpolator(sitk.sitkLinear)

    if kind == "nonlinear":
        # Must match the mesh the warp was made with, or this under-fits by
        # construction and no amount of optimising will close the gap.
        mesh = mesh or DEFAULTS["nonlinear"]["mesh"]
        transform = sitk.BSplineTransformInitializer(f, list(mesh))
        method.SetOptimizerAsLBFGSB(numberOfIterations=iterations)
    elif kind in KINDS:
        base = {
            "rigid": sitk.Euler2DTransform(),
            "similarity": sitk.Similarity2DTransform(),
            "affine": sitk.AffineTransform(2),
        }[kind]
        # Centres the transform on the images; without it rotation happens about
        # the corner and every rotation looks like a huge translation.
        transform = sitk.CenteredTransformInitializer(
            f, m, base, sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=4.0, minStep=1e-4, numberOfIterations=iterations
        )
        method.SetOptimizerScalesFromPhysicalShift()
    else:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    # ponytail: single resolution level, so the animation reads as one clean
    # descent. Add SetShrinkFactorsPerLevel to demo coarse-to-fine.
    method.SetInitialTransform(transform, inPlace=True)

    history = []
    if capture:
        method.AddCommand(
            sitk.sitkIterationEvent,
            lambda: history.append(
                Step(
                    method.GetOptimizerIteration(),
                    method.GetMetricValue(),
                    _resample(m, f, transform),
                )
            ),
        )

    final = method.Execute(f, m)
    return Registration(
        _resample(m, f, final),
        final,
        history,
        method.GetOptimizerStopConditionDescription(),
    )


def transform_error(truth, recovered, image, step=8):
    """Mean residual displacement, in pixels, over a grid of points.

    `truth` and `recovered` should compose to the identity for a perfect
    recovery, so this measures how far each point ends up from where it
    started. Works for every transform kind — B-spline control points aren't
    unique, so comparing parameters directly would not.
    """
    img = _as_image(image)
    width, height = img.GetSize()
    points = [
        (float(x), float(y))
        for y in range(0, height, step)
        for x in range(0, width, step)
    ]
    residuals = [
        np.subtract(truth.TransformPoint(recovered.TransformPoint(p)), p)
        for p in points
    ]
    return float(np.linalg.norm(residuals, axis=1).mean())


def recover(image, kind="rigid", register_with=None, capture=True, **params):
    """Deform an image, register it back, and report the error in pixels.

    `register_with` defaults to `kind` — set it to something less flexible
    (e.g. deform "nonlinear", register "rigid") to demo under-fitting.
    """
    truth = deform(image, kind, **params)
    reg = register(
        image,
        truth.image,
        register_with or kind,
        capture=capture,
        mesh=params.get("mesh"),
    )
    return Recovery(
        image=reg.image,
        error=transform_error(truth.transform, reg.transform, image),
        deformed=truth.image,
        truth=truth,
        registration=reg,
    )


def _norm(a):
    a = np.asarray(a, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def overlay(fixed, moving, alpha=0.55):
    """Fixed image in green, moving in magenta. Aligned structures go grey.

    `alpha` is how strongly the fixed image tints the result. Lower values keep
    the moving image readable while it slides into place — but only the
    *disagreement* fades, because where the two match the green channel equals
    the moving image regardless of alpha, so aligned regions still go grey.
    """
    f, m = _norm(fixed), _norm(moving)
    return np.dstack([m, alpha * f + (1.0 - alpha) * m, m])


def _caption(step, label=None):
    line = f"iteration {step.iteration} — loss {step.metric:,.0f}"
    return f"{label}\n{line}" if label else line


def animate(
    history, fixed, path=None, fps=4, max_frames=60, dpi=80, label=None, alpha=0.55
):
    """Animate a captured registration: overlay on the left, loss curve on the right.

    The loss curve shows every iteration; the frames are thinned to at most
    `max_frames` so a 400-iteration run still makes a watchable GIF. Pass
    `path` to write a GIF (or an .mp4, if ffmpeg is around).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not history:
        raise ValueError("empty history — call register(..., capture=True)")

    iterations = [s.iteration for s in history]
    losses = [s.metric for s in history]
    # Weighted toward early iterations, where the loss actually moves. Linear
    # spacing spends most of the GIF on a near-static tail; squaring shows the
    # opening 1:1 and subsamples the flat part hard.
    frames = np.unique(
        (np.linspace(0, 1, max_frames) ** 2 * (len(history) - 1)).astype(int)
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.2))

    picture = left.imshow(overlay(fixed, history[0].image, alpha))
    left.axis("off")
    # Real text before tight_layout, or it reserves no room and clips the title.
    caption = left.set_title(_caption(history[0], label), fontsize=10)

    right.plot(iterations, losses, color="0.75")
    marker, = right.plot([iterations[0]], [losses[0]], "o", color="crimson")
    right.set_xlabel("iteration")
    right.set_ylabel("loss (mean squares)")
    # From zero, so an under-fit plateau reads as "stuck high" rather than as a
    # noisy curve that autoscale has zoomed into.
    right.set_ylim(bottom=0)
    right.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    def update(i):
        step = history[i]
        picture.set_data(overlay(fixed, step.image, alpha))
        marker.set_data([step.iteration], [step.metric])
        caption.set_text(_caption(step, label))
        return picture, marker, caption

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    if path:
        anim.save(path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)
    return anim


def _sample_image():
    """A naturalistic photo — texture everywhere, and obvious when it's wrong.

    Bundled with scikit-image, so this needs no network. Halved to 256 px to
    keep the registrations quick enough to re-run live in a lecture.
    """
    from skimage.data import camera

    return camera()[::2, ::2].astype(np.float32)


def demo():
    """Round-trip every transform kind and check the error is sub-pixel."""
    image = _sample_image()

    # Pin the two conventions the docstrings claim: `translation` describes how
    # the picture visibly moves, and `field` is the pull transform's, (x, y) last.
    pinned = deform(image, "rigid", angle=0.0, translation=(20.0, 0.0))
    assert np.allclose(pinned.field, [-20.0, 0.0], atol=1e-3), "field convention"
    assert np.allclose(pinned.image[:, 20:], image[:, :-20], atol=1e-3), "shift sign"

    for kind in KINDS:
        result = recover(image, kind, capture=False)
        # B-spline warps are looser: the recovered field only has to agree where
        # there is texture to constrain it.
        # A regression guard, not a quality bar. A warp this size pushes some
        # corners out of frame, where nothing constrains the fit, and those
        # unconstrained points inflate the mean well past what the eye sees.
        limit = 7.0 if kind == "nonlinear" else 1.0
        assert result.error < limit, f"{kind}: {result.error:.2f} px > {limit} px"
        stop = result.registration.stop
        assert "Maximum number of iterations" not in stop, f"{kind} gave up: {stop}"
        print(f"{kind:>10}: recovered to {result.error:.2f} px — {stop}")

    # A finer mesh must reach `register` too, or recover() under-fits silently.
    # Absolute error is the wrong test here — a 10x10 warp is hard whatever you
    # do — so check that matching the mesh helps, and that recover() gets there.
    warp = deform(image, "nonlinear", mesh=(10, 10))

    def warp_error(reg):
        return transform_error(warp.transform, reg.transform, image)

    matched = warp_error(register(image, warp.image, "nonlinear", mesh=(10, 10)))
    coarse = warp_error(register(image, warp.image, "nonlinear"))
    forwarded = recover(image, "nonlinear", mesh=(10, 10), capture=False).error
    assert matched < coarse, f"finer mesh did not help: {matched:.2f} vs {coarse:.2f}"
    assert abs(forwarded - matched) < 1e-6, "recover() did not forward mesh"
    print(f" 10x10 mesh: {matched:.2f} px, vs {coarse:.2f} px at the coarse default")

    # Under-fitting: a rigid transform cannot undo a non-linear warp.
    underfit = recover(image, "nonlinear", register_with="rigid", capture=False)
    assert underfit.error > 1.0, f"rigid unexpectedly undid a warp: {underfit.error}"
    print(f"underfit (nonlinear warp, rigid fit): {underfit.error:.2f} px")


if __name__ == "__main__":
    demo()

    image = _sample_image()
    for kind in KINDS:
        result = recover(image, kind)
        history = result.registration.history
        animate(history, image, path=f"img/register-{kind}.gif")
        print(f"wrote img/register-{kind}.gif ({len(history)} optimiser steps)")

    # Under-fitting: the transform is too rigid to express the deformation, so
    # the loss flattens out early while the picture is still visibly wrong.
    underfits = [
        # Pure shear plus a modest rotation: rigid can undo the rotation and
        # nothing else, so you watch it fix half the problem and stall.
        ("affine", "rigid", dict(shear=0.4, scale=1.0, angle=10.0, translation=(12.0, -8.0))),
        ("nonlinear", "affine", {}),
    ]
    for deformed_by, fitted_with, params in underfits:
        result = recover(image, deformed_by, register_with=fitted_with, **params)
        history = result.registration.history
        name = f"img/underfit-{fitted_with}-on-{deformed_by}.gif"
        animate(
            history,
            image,
            path=name,
            label=f"{fitted_with} fit → {deformed_by} deformation"
            f" ({result.error:.1f} px off)",
        )
        print(f"wrote {name} ({result.error:.1f} px residual)")
