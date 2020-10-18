//
// Created by g21589 on 10/2/20.
//
#include "Base.h"
#include "base/bind.h"
#include "base/command_line.h"
#include "base/strings/string_util.h"
#include "base/strings/stringprintf.h"
#include "headless/public/devtools/domains/runtime.h"
#include "../utils.h"
#include <streambuf>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#ifndef CHROMIUM_SUPERVIZSCOPE_H
#define CHROMIUM_SUPERVIZSCOPE_H

namespace kaleido {
    namespace scopes {

        class SupervizScope : public BaseScope {
        public:
            SupervizScope();

            ~SupervizScope() override;

            SupervizScope(const SupervizScope &v);

            std::string ScopeName() override;

            std::vector<std::unique_ptr<::headless::runtime::CallArgument>> BuildCallArguments() override;

        };

        SupervizScope::SupervizScope() {
            // Add MathJax config
            //scriptTags.emplace_back("window.PlotlyConfig = {MathJaxConfig: 'local'}");

            // Process customjs
            if (HasCommandLineSwitch("customjs")) {
                std::string customjsArg = GetCommandLineSwitch("customjs");

                // Check if value is a URL
                GURL customjsUrl(customjsArg);
                if (customjsUrl.is_valid()) {
                    scriptTags.push_back(customjsArg);
                } else {
                    // Check if this is a local file path
                    if (std::ifstream(customjsArg)) {
                        localScriptFiles.emplace_back(customjsArg);
                    } else {
                        errorMessage = base::StringPrintf("--customjs argument is not a valid URL or file path: %s",
                                                          customjsArg.c_str());
                        return;
                    }
                }
            } else {
                // TODO: custom.js path
                scriptTags.emplace_back("custom.js");
            }

        }

        SupervizScope::~SupervizScope() {}

        SupervizScope::SupervizScope(const SupervizScope &v) {}

        std::string SupervizScope::ScopeName() {
            return "superviz";
        }

        std::vector<std::unique_ptr<::headless::runtime::CallArgument>> SupervizScope::BuildCallArguments() {
            std::vector<std::unique_ptr<::headless::runtime::CallArgument>> args;
            return args;
        }
    }
}

#endif //CHROMIUM_SUPERVIZSCOPE_H
