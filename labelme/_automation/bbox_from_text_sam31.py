from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._sam31_session import Sam31Session


def get_bboxes_from_texts(
    session: Sam31Session,
    image: NDArray[np.uint8],
    image_id: str,
    texts: list[str],
    min_score: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32], None]:
    boxes, scores, labels = session.run(
        image=image,
        image_id=image_id,
        texts=texts,
        min_score=min_score,
    )
    return boxes, scores, labels, None
