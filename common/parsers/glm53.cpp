#include "parsers.h"

// GLM-5.3-Flash - forced-open <think> and tagged tool calls:
//   <|assistant|><think> [reasoning] [</think>] [content]
//   {<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>}
// The generation prompt is "<|assistant|><think>", so the model never emits the
// opening think tag. A tool call may arrive before </think>. The template tells
// the model to emit one or more calls, so every complete call in the turn is
// parsed even when the client omits parallel_tool_calls.
common_chat_params common_chat_params_init_glm53(const common_chat_template &          tmpl,
                                                        const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    const std::string ASSISTANT  = "<|assistant|>";
    const std::string THINK_START = "<think>";
    const std::string THINK_END   = "</think>";
    const std::string CALL_START  = "<tool_call>";
    const std::string CALL_END    = "</tool_call>";
    const std::string ARG_KEY     = "<arg_key>";
    const std::string ARG_KEY_END = "</arg_key>";
    const std::string ARG_VAL     = "<arg_value>";
    const std::string ARG_VAL_END = "</arg_value>";

    data.preserved_tokens = {
        THINK_START, THINK_END, CALL_START, CALL_END,
        ARG_KEY, ARG_KEY_END, ARG_VAL, ARG_VAL_END,
    };

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = { THINK_END, CALL_START };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, ASSISTANT },
        { COMMON_CHAT_ROLE_USER,      "<|user|>" },
        { COMMON_CHAT_ROLE_TOOL,      "<|observation|>" },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|system|>" },
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = ASSISTANT + THINK_START + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    bool think_open = false;
    if (inputs.has_continuation()) {
        think_open = inputs.continue_final_message != COMMON_CHAT_CONTINUATION_CONTENT;
    } else {
        auto last_open  = data.generation_prompt.rfind(THINK_START);
        auto last_close = data.generation_prompt.rfind(THINK_END);
        think_open      = last_open != std::string::npos &&
                     (last_close == std::string::npos || last_open > last_close);
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto end = p.end();

        auto opener = p.optional(p.literal(ASSISTANT) + p.optional(p.space()));

        auto body_end   = think_open ? p.until_one_of({ THINK_END, CALL_START }) : p.until_one_of({ THINK_END });
        auto think_body = extract_reasoning ? p.reasoning(body_end) : p.content(body_end);

        auto reasoning = p.optional(p.optional(p.literal(THINK_START)) + think_body +
                                    p.optional(p.literal(THINK_END)));

        auto content = p.optional(p.content(p.until_one_of({ CALL_START })));

        if (has_response_format) {
            auto closed_reasoning = p.literal(THINK_START) + think_body + p.literal(THINK_END);
            auto response_format  = p.content(p.schema(p.json(), "response-format", inputs.json_schema));
            return opener + (closed_reasoning << response_format) + end;
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return opener + reasoning + p.optional(p.content(p.rest())) + end;
        }

        auto tool_choices = p.choice();
        auto arg_close    = p.tool_arg_close(p.literal(ARG_VAL_END));
        auto arg_string   = p.rule("glm53-arg-string",
                                   p.tool_arg_string_value(p.until(ARG_VAL_END)) + arg_close);

        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");

            std::vector<common_peg_parser> required_args;
            std::vector<common_peg_parser> optional_args;

            foreach_parameter(function, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                auto rule_name = "glm53-arg-" + name + "-" + param.name;
                auto types = param.schema->value_types();

                // String values are raw text up to the closing tag. Other types
                // are JSON. A union tries JSON first so numbers and nulls stay typed.
                auto arg_value = p.eps();
                if (!types.has(common_chat_schema::TYPE_STRING)) {
                    arg_value = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) + arg_close;
                } else if (types.is_only(common_chat_schema::TYPE_STRING)) {
                    arg_value = arg_string;
                } else {
                    arg_value = p.gbnf(p.atomic(p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) + arg_close) | arg_string,
                                       "glm53-arg-string");
                }

                auto arg = p.rule(rule_name,
                    p.optional(p.space()) +
                    p.tool_arg(p.tool_arg_open(p.literal(ARG_KEY) + p.tool_arg_name(p.literal(param.name)) +
                                               p.literal(ARG_KEY_END)) +
                               p.optional(p.space()) + p.literal(ARG_VAL) +
                               arg_value));

                (param.required ? required_args : optional_args).push_back(arg);
            });

            auto args = p.permute("glm53-" + name + "-args", required_args);
            if (!optional_args.empty()) {
                args = args + p.zero_or_more(p.choice(optional_args));
            }

            // Do not commit a name that is a prefix of another tool. special_function
            // must fail on special_function_with_opt instead of opening a call that
            // the next character then retracts.
            auto name_boundary = p.peek(p.optional(p.space()) + (p.literal(ARG_KEY) | p.literal(CALL_END)));
            auto call = p.tool(p.tool_open(p.atomic(p.literal(CALL_START) + p.tool_name(p.literal(name)) + name_boundary) +
                                           p.optional(p.space())) +
                               p.tool_args(args) +
                               p.tool_close(p.optional(p.space()) + p.literal(CALL_END)));

            tool_choices |= p.rule("glm53-tool-" + name, call);
        });

        // The chat template says "one or more functions". A later call must not
        // fall through into content when parallel_tool_calls is omitted or false.
        auto calls = tool_choices + p.zero_or_more(p.space() + tool_choices);

        auto tools_section = p.trigger_rule("glm53-tool-call", calls + p.optional(p.content(p.rest())));

        auto tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? tools_section :
                                                                             p.optional(tools_section);

        return opener + reasoning + content + tools + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, CALL_START },
        };
    }

    return data;
}
