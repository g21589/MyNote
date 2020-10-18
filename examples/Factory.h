//
// Created by jmmease on 6/17/20.
//

#include "base/strings/string_util.h"

#include "Plotly.h"
#include "Superviz.h"
#include "Base.h"

#ifndef CHROMIUM_FACTORY_H
#define CHROMIUM_FACTORY_H

kaleido::scopes::BaseScope* LoadScope(std::string name) {
    std::string name_lower = base::ToLowerASCII(name);
    if (name_lower == "plotly") {
        return new kaleido::scopes::PlotlyScope();
    } else if (name_lower == "superviz") {
        return new kaleido::scopes::SupervizScope();
    } else {
        return nullptr;
    }
}

#endif //CHROMIUM_FACTORY_H
