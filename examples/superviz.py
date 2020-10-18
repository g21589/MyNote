from __future__ import absolute_import
from kaleido.scopes.base import BaseScope
import base64


class SupervizScope(BaseScope):
    """
    Scope for transforming Custom figures to static images
    """
    _all_formats = ("png", "jpg", "jpeg", "webp", "svg", "pdf", "eps", "json")
    _text_formats = ("svg", "json", "eps")

    _scope_flags = ("customjs",)

    def __init__(self, customjs=None, **kwargs):
        # Save scope flags as internal properties
        self._customjs = customjs

        # to_image-level default values
        self.default_format = "png"
        self.default_width = 700
        self.default_height = 500
        self.default_scale = 1

        super(SupervizScope, self).__init__(**kwargs)

    @property
    def scope_name(self):
        return "superviz"

    # Flag property methods
    @property
    def customjs(self):
        """
        URL or local file path to custom.js bundle to use for image export.
        If not specified, will default to CDN location.
        """
        return self._customjs

    @customjs.setter
    def customjs(self, val):
        self._customjs = val
        self._shutdown_kaleido()
