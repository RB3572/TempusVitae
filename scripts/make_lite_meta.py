#!/usr/bin/env python3
"""Strip every accuracy claim out of the lightweight model's metadata.

WHY THIS EXISTS. The ViT-B trunk this build is cut from (`cache/ssl_base`) was
pretrained before the vault was frozen, so it has seen the sealed sessions -- see
HANDOFF.md rule 3, "`ssl_base` has seen tier 2 and cannot be scored on it". Every
cross-validated figure for it is therefore optimistic by an unknown amount, and none of
it is comparable to the published ViT-L numbers. The decision was to ship the model and
claim nothing about its accuracy.

Nothing in the site renders these fields today. They are removed anyway, because the
failure mode is somebody later writing `meta.valMae` into the page and shipping a number
that was never earned.

    python scripts/make_lite_meta.py ../cache/lite_export/model_meta.json \
        public/models/mouse_lite.meta.json
"""

import json
import sys
from pathlib import Path

# Anything that asserts how well the model does, or on what it was measured.
ACCURACY_FIELDS = [
    "valMae", "valMaeCI95", "valRhoWithin", "valSlope", "valProtocol",
    "vaultMae", "externalMae", "externalRhoWithin",
    "baseline", "nEmbryos", "nFrames", "evalStandard", "notAdopted",
]


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads(src.read_text())
    dropped = [k for k in ACCURACY_FIELDS if k in meta]
    for k in dropped:
        meta.pop(k)
    # The recipe string is read from the shared analysis record and describes the ViT-L:
    # "frozen DINOv2 ViT-L/14 + temporal SSL, TTA-8, 3-seed head". It is WRONG here on
    # both the trunk and the view count, and unlike the accuracy fields it IS rendered --
    # it goes into the PDF export's header. Rewritten from the bundle's own facts.
    meta["recipe"] = (
        f"frozen DINOv2 ViT-B/14 + temporal SSL, "
        f"{meta.get('ttaViews', 1)} view, 3-seed head"
    )
    meta["accuracyWithheld"] = (
        "No accuracy figure is published for this build. Its trunk saw the sealed "
        "sessions during self-supervised pretraining (HANDOFF.md rule 3), so every "
        "cross-validated number for it is optimistic by an unknown amount and none of "
        "them describe this artifact the way the ViT-L figures describe the full model."
    )
    dst.write_text(json.dumps(meta, indent=2))
    print(f"  dropped: {', '.join(dropped) if dropped else '(none present)'}")
    print(f"  wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
