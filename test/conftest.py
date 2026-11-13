from collections.abc import Callable
from typing import cast

import pytest

from wound_composite.layout import Layout, templates


@pytest.fixture(params=templates.__all__)
def layout(request: pytest.FixtureRequest) -> Layout:
    return cast(Callable[[], Layout], getattr(templates, request.param))()
