# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import re
import sys
import logging

"""
    generate stub func body by return type
"""
RETURN_STATEMENTS = {
    'std::string': '    return "";',
    'ShapeVector': '    op::ShapeVector ret;\n'
                   '    return ret;',
    'PlatformInfo& PlatformInfo::GetCurrentPlatformInfo': '    static PlatformInfo platform;\n'
                                                          '    return platform;',
    'SocVersion': '    return SocVersion::RESERVED_VERSION;',
    'int64_t': '    return 0;',
    'PlatformInfo&': '    static PlatformInfo platform;\n'
                     '    return platform;',
    'Ptr': '    return nullptr;',
    'aclnnStatus': '    return ACLNN_SUCCESS;',
    'OpImplMode': '    return OpImplMode::IMPL_MODE_RESERVED;',
    'std::string&': '    static std::string s;\n'
                    '    return s;',
    'aclrtStream': '    aclrtStream stream{nullptr};\n'
                   '    return stream;',
    'Format': '    return Format::FORMAT_ND;',
    'DataType': '    return DataType::DT_FLOAT;',
    'int32_t': '    return 0;',
    'uint16_t': '    return 0;',
    'int16_t': '    return 0;',
    'uint8_t': '    return 0;',
    'int8_t': '    return 0;',
    'double': '    return 0;',
    'uint64_t': '    return 0;',
    'wchar_t': '    return 0;',
    'uint32_t': '    return 0;',
    'size_t': '    return 0;',
    'bool': '    return false;',
    'float': '    return 0;',
    'int': '    return 0;',
    'T': '    return 0;',
    'op::fp16_t': '    return 0;',
    'op::bfloat16': '    return 0;',
    'complex': '    return 0;',
    'aclStorage::aclStorage': '',
    'op::Format': '    return op::Format::FORMAT_ND;',
    'op::Shape&': '    static op::Shape s;\n'
                  '    return s;',
    'op::DataType': '    return op::DataType::DT_FLOAT;',
    'aclTensor*&': '    static aclTensor* a{nullptr};\n'
                   '    return a;',
    'op::TensorPlacement': '    return op::TensorPlacement::kOnDeviceHbm;', 
    'op::Strides&': '    static op::Strides s;\n'
                    '    return s;',
}

"""
    white_list_for_debug, include_dir_key_words is to
    determines which header files to generate cc files from
    when DEBUG on
"""
white_list_for_debug = [
    "tensor_view_utils.h",
    "shape_utils.h",
    "platform.h",
    "op_executor.h",
    "op_dfx.h",
    "op_def.h",
    "framework_op.h",
    "format_utils.h",
    "data_type_utils.h",
    "common_types.h",
    "aicpu/aicpu_task.h", # aicpu
    "individual_op_api.h",  # nnopbase
    "aclnn/acl_meta.h",
    "bridge_pool.h",
    "block_pool.h",
    "aicpu/aicpu_args_handler.h", # aicpu
    "fp16_t.h",
    "op_log.h",
    "block_store.h",
    "op_cache.h",
    "op_arg_def.h",
    "pool_allocator.h",
    "object.h"
]
"""
    spec_list_for_debug, add specific content when DEBUG on
"""
spec_list_for_debug = [
    {"block_pool.h": "BlockPool globalPoolImpl__  __attribute__ ((init_priority (200)));\n\
        BlockPool &BlockPool::globalPool_ = globalPoolImpl__;\n"}
]

include_dir_key_words = ["include/nnopbase/opdev/", "include/nnopbase/aclnn/",
    "src/nnopbase/common/inc", "src/nnopbase/individual_op/api/",
    "src/nnopbase/composite_op/aclnn_engine/", "src/nnopbase/composite_op/mem_mgr/"]
"""
    this attr is used for symbol table visible
"""

"""
    max code len per line in hua_wei software programming specifications
"""
MAX_CODE_LEN_PER_LINE = 100

DEBUG = True

logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] [%(lineno)s] %(levelname)s: %(message)s',
                    level=logging.INFO)


def need_generate_func(func_line, file_path):
    """
    :param func_line:
    :return:
    """
    file_name = file_path.split("/")[-1]
    func_compact = re.sub(r'\s+', '', func_line.strip())
    if func_compact.endswith(re.sub(r'\s+', '', "aclTensor::aclTensor(const T *value, uint64_t size, op::DataType dataType)")):
        return False
    if file_name == "common_types.h":
        if func_compact == re.sub(r'\s+', '', 'size_t TypeSize(DataType dataType)'):
            return False
        if func_compact == re.sub(r'\s+', '', 'ge::AscendString ToString(op::DataType dataType)'):
            return False
        if func_compact == re.sub(r'\s+', '', 'void ToContiguousStrides(const op::Shape &shape, op::Strides &strides)'):
            return False
    if func_line.strip().endswith("default") or func_line.strip().endswith("delete") \
            or func_line.strip().startswith("typedef") or func_line.strip().startswith("using"):
        return False
    return True


def file_endswith_white_list_suffix(file):
    """
    :param file:
    :return:
    """
    if DEBUG:
        for suffix in white_list_for_debug:
            suffix = re.sub(r'^/*', '/', suffix)
            if file.endswith(suffix):
                return True
        return False
    else:
        return True


"""
    belows are patterns used for analyse .h file
"""
# pattern function
pattern_func = re.compile(r"""(\[\[maybe_unused\]\]\s+)?([\s]*)([a-zA-Z~_].*[)](?!.*{).*)(;.*)\n$""",
    re.VERBOSE | re.MULTILINE | re.DOTALL)
# pattern virtual function
pattern_virtual_func = re.compile(r"""^[ ]*
                                       virtual
                                       [ ]+
                                       (?:const[ ]+)?
                                       [:\w]+
                                       [ &*]+
                                       [:\w]+
                                       \(
                                         [^()]*
                                       \)
                                       [ ]+
                                       (?:const[ ]+)?
                                       =[ ]+0;$""", re.VERBOSE)
                           
# pattern comment
pattern_comment = re.compile(r'^\s*//')
pattern_comment_2_start = re.compile(r'^\s*/[*]')
pattern_comment_2_end = re.compile(r'[*]/\s*$')
pattern_comment_special_end = re.compile('},')
# pattern define
pattern_define = re.compile(r'^\s*#define')
pattern_define_return = re.compile(r'\\\s*$')
# pattern format_error_message
pattern_format_error_message = re.compile(r'^int32_t FormatErrorMessage')
pattern_format_error_message_return = re.compile(r'__attribute__')
# blank line
pattern_blank_line = re.compile(r'^\s*$')
# virtual,explicit,friend,static
pattern_keyword = re.compile(r'(virtual\s+|explicit\s+|friend\s+|static\s+)')
# lead space
pattern_leading_space = re.compile(r'(^[\s]*)[a-zA-Z~_]')
# functions will have patterns such as func ( or func(
# but operator is an exception; the class name is preceded by an operator, and the above mode does not exist
# format like :"operator = ()"
pattern_func_name = re.compile(r'([a-zA-Z0-9~_\-]+\s*|operator?.*)[(]')
# template
pattern_template = re.compile(r'^\s*template')
pattern_template_end = re.compile(r'>\s*$')
# namespace
pattern_namespace = re.compile(r'namespace.*{')
# extern C
pattern_externc = re.compile(r'extern.*{')
# global var
pattern_gvar = re.compile(r'extern\s+(thread_local\s+)?(\w+)\s+(\w+)(\[\w+\])?;')
# class : which can handle classA a and {not on the same line, but if found ';' after class,then don't deal with
pattern_class = re.compile(r'^[\s]*(class|struct|typedef struct)\s+(%s\s+)?([a-zA-Z0-9_\-]+<?)(?!.*;)')
# pattern for function body start and end
pattern_start = re.compile('{')
pattern_end = re.compile('}')
# pattern for ACL_ARRAY
pattern_acl_arrary = re.compile('ACL_ARRAY')
pattern_empty_op_arg = re.compile('EMPTY_OP_ARG')
pattern_constructor = re.compile('::')
pattern_aclnn_with_binary = re.compile('ACLNN_WITH_BINARY')
pattern_macro_else = re.compile('^\s*#else')

# pattern for format func
pat_format_func = re.compile(r"""^(
                                   (?:const[ ]+)?
                                   (?:
                                    [:\w]+
                                    |
                                    std::(?:vector|shared_ptr)<[:\w ]+>
                                    |
                                    std::(?:vector|shared_ptr)<std::vector<[:\w ]+>>
                                    |
                                    std::(?:map|unordered_map|pair)<[:\w]+[, ]+[:\w]+>
                                   )
                                  )
                                  ([ ]+)
                                  ([&*]+)""", re.VERBOSE)
# pattern for parser ret_type & func_name
pat_search_func = re.compile(r"""^(?:const[ ]+)?
                                  (?P<ret_type>
                                   (?:
                                    [:\w]+
                                    |
                                    std::(?:vector|shared_ptr)<[:\w ]+>
                                    |
                                    std::(?:vector|shared_ptr)<std::vector<[:\w ]+>>
                                    |
                                    std::(?:map|unordered_map|pair)<[:\w]+[, ]+[:\w]+>
                                   )
                                   (?:[&*]+)?
                                  )
                                  [ ]+
                                  (?P<class_name>\w+)
                                  ::
                                  \n?
                                  (?P<func_name>\w+|operator=)
                                  [ ]*
                                  \(""", re.VERBOSE)


class H2CC(object):
    def __init__(self, input_file, output_file, shared_includes_content):
        """
        :param input_file:
        :param output_file:
        :param shared_includes_content:
        """
        self.input_file = input_file
        self.output_file = output_file
        self.shared_includes_content = shared_includes_content
        self.line_index = 0
        self.input_fd = open(self.input_file, 'r')
        self.input_content = self.input_fd.readlines()
        self.output_fd = open(self.output_file, 'w')

        # The state may be normal_now(in the middle of {}),class_now,namespace_now
        self.stack = []
        self.stack_class = []
        self.stack_template = []
        # record funcs generated by h2cc func
        self.func_list_exist = []

    def __del__(self):
        self.input_fd.close()
        self.output_fd.close()
        del self.stack
        del self.stack_class
        del self.stack_template
        del self.func_list_exist

    def just_skip(self):
        # skip blank line or comment
        if pattern_blank_line.search(self.input_content[self.line_index]) or pattern_comment.search(
                self.input_content[self.line_index]):  # /n or comment using //
            self.line_index += 1
            return True
        if pattern_comment_2_start.search(self.input_content[self.line_index]):  # comment using /*
            while (not pattern_comment_2_end.search(self.input_content[self.line_index]) and 
                not pattern_comment_special_end.search(self.input_content[self.line_index])):
                self.line_index += 1
            self.line_index += 1
            return True
        # skip define
        if pattern_define.search(self.input_content[self.line_index]):
            while pattern_blank_line.search(self.input_content[self.line_index]) or pattern_define_return.search(
                    self.input_content[self.line_index]):
                self.line_index += 1
            self.line_index += 1
            return True
        # skip virtual function
        while pattern_virtual_func.search(self.input_content[self.line_index]):
            self.line_index += 1
            return True
        while pattern_acl_arrary.search(self.input_content[self.line_index]):
            self.line_index += 1
            return True
        while pattern_empty_op_arg.search(self.input_content[self.line_index]):
            self.line_index += 1
            return True
        if pattern_aclnn_with_binary.search(self.input_content[self.line_index]):
            self.line_index += 1
            while not pattern_macro_else.search(self.input_content[self.line_index]):
                self.line_index += 1
            return True
        return False

    def write_inc_content(self):
        self.shared_includes_content.sort()
        for shared_include_content in self.shared_includes_content:
            self.output_fd.write(shared_include_content)

    def write_specified_template(self):
        if self.input_file.find("common_types.h") == -1:
            return
        content = '''
        template class aclArray<int64_t>;
        template class aclArray<float>;
        template class aclArray<bool>;
        template class aclArray<op::fp16_t>;
        template class aclArray<op::bfloat16>;

        template<typename T>
        aclTensor::aclTensor(const T *value, uint64_t size, op::DataType dataType)
        {
        }
        template aclTensor::aclTensor(const int64_t *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const uint64_t *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const int32_t *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const uint32_t *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const int8_t *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const uint8_t *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const int16_t *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const uint16_t *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const float *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const double *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const bool *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const char *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const op::bfloat16 *value, uint64_t size, op::DataType dataType);
        template aclTensor::aclTensor(const op::fp16_t *value, uint64_t size, op::DataType dataType);

        template bool aclScalar::CheckOverflows<bool>() const;
        template bool aclScalar::CheckOverflows<int8_t>() const;
        template bool aclScalar::CheckOverflows<uint8_t>() const;
        template bool aclScalar::CheckOverflows<int16_t>() const;
        template bool aclScalar::CheckOverflows<uint16_t>() const;
        template bool aclScalar::CheckOverflows<int32_t>() const;
        template bool aclScalar::CheckOverflows<uint32_t>() const;
        template bool aclScalar::CheckOverflows<int64_t>() const;
        template bool aclScalar::CheckOverflows<uint64_t>() const;
        template bool aclScalar::CheckOverflows<op::fp16_t>() const;
        template bool aclScalar::CheckOverflows<op::bfloat16>() const;
        template bool aclScalar::CheckOverflows<float>() const;
        template bool aclScalar::CheckOverflows<double>() const;
        template bool aclScalar::CheckOverflows<std::complex<float>>() const;
        template bool aclScalar::CheckOverflows<std::complex<double>>() const;

        template void aclTensor::SetData<float>(int64_t index, const float value, op::DataType dataType);
        '''
        self.output_fd.write(content)

    def h2cc(self):
        """
        :return:
        """
        logging.info("start generate cc_file[%s] from h_file[%s]", self.output_file, self.input_file)
        # write inc content
        self.write_inc_content()
        # core processing cycle, process the input .h file by line
        while self.line_index < len(self.input_content):
            # handle comment and blank line
            if (self.just_skip()):
                continue

            # match namespace
            self.handle_namespace()
            self.handle_externc()
            self.handle_global_var()

            # match template
            template_string = self.handle_template()
            # match class
            line = self.input_content[self.line_index]
            logging.info("line[%s]", line)
            match_class = pattern_class.search(line)
            match_start = pattern_start.search(line)
            handle_class_result = self.handle_class(template_string, line, match_start, match_class)
            if handle_class_result == "continue":
                continue

            # match "}"
            handle_stack_result = self.handle_stack(match_start)
            if handle_stack_result == "continue":
                continue

            # handle func
            handle_func1_result, line, start_i = self.handle_func1(line, template_string)
            if handle_func1_result == "continue":
                continue

            if "op_log.h" in self.input_file:
                if "DlogInner" in line or "DlogWrite" in line or "DlogRecord" in line:
                    self.line_index += 1
                    continue

            # here means func is found
            # delete key word
            contain_friend = re.search('friend ', line)
            line = pattern_keyword.sub('', line)
            logging.info("line[%s]", line)

            # Class member function
            # if friend we will not add class name
            friend_match = re.search('friend ', line)
            if len(self.stack_class) > 0 and not friend_match:
                line, func_name = self.handle_class_member_func(line, template_string)
            # Normal functions
            else:
                line, func_name = self.handle_normal_func(line, template_string)
            need_generate = need_generate_func(line, self.input_file)
            # func body
            func_content = line
            func_content = re.sub('override$', '', func_content)
            func_def = self.implement_function(line)
            line = func_content + func_def
            # comment
            line = self.gen_comment(start_i) + line

            # write to out file
            if not contain_friend:
                for black_file in spec_list_for_debug:
                    for key, val in black_file.items():
                        if key in self.input_file and val not in line:
                            line = line + val
                            spec_list_for_debug.remove(black_file)
                self.write_func_content(line, func_name, need_generate)

            # next loop
            self.line_index += 1
        self.write_specified_template()
        logging.info('Added %s functions', len(self.func_list_exist))
        logging.info('Successfully converted,please see ' + self.output_file)

    def handle_func1(self, line, template_string=''):
        """
        :param line:
        :return:
        """
        find1 = re.search('[(]', line)
        if not find1:
            self.line_index += 1
            return "continue", line, None
        find2 = re.search('[)]', line)
        start_i = self.line_index
        space_match = pattern_leading_space.search(line)
        # deal with
        # int abc(int a,
        #        int b)
        if find1 and (not find2):
            self.line_index += 1
            line2 = self.input_content[self.line_index]
            if space_match:
                line2 = re.sub('^' + space_match.group(1), '', line2)
            line += line2
            while self.line_index < len(self.input_content) and (not re.search('[)]', line2)):
                self.line_index += 1
                line2 = self.input_content[self.line_index]
                line2 = re.sub('^' + space_match.group(1), '', line2)
                line += line2

        match_start = pattern_start.search(self.input_content[self.line_index])
        match_end = pattern_end.search(self.input_content[self.line_index])
        if match_start:  # like  ) {  or ) {}    int the last line
            if not match_end:
                self.stack.append('normal_now')
            ii = start_i
            while ii <= self.line_index:
                ii += 1
            self.line_index += 1
            return "continue", line, start_i
        logging.info("line[%s]", line)
        # '  int abc();'->'int abc()'
        (line, match) = pattern_func.subn(r'\3\n', line)
        logging.info("line[%s]", line)
        # deal with case of 'return type' and 'func_name' are not in the same line, like: 'int \n abc(int a, int b)'
        prev_line = self.input_content[start_i - 1]
        stripped_prev = prev_line.strip()
        if (not template_string and stripped_prev
                and not stripped_prev.endswith(':')
                and re.match(r'^[\w:<>*&\s]+$', stripped_prev)):
            line = prev_line + line
        line = line.lstrip()
        if not match:
            self.line_index += 1
            return "continue", line, start_i
        return "pass", line, start_i

    def handle_stack(self, match_start):
        """
        :param match_start:
        :return:
        """
        line = self.input_content[self.line_index]
        match_end = pattern_end.search(line)
        if match_start:
            self.stack.append('normal_now')
        if match_end:
            top_status = self.stack.pop()
            if top_status == 'namespace_now':
                self.output_fd.write(line + '\n')
            elif top_status == 'externc_now':
                self.output_fd.write(line + '\n')                
            elif top_status == 'class_now':
                self.stack_class.pop()
                self.stack_template.pop()
        if match_start or match_end:
            self.line_index += 1
            return "continue"
        if len(self.stack) > 0 and self.stack[-1] == 'normal_now':
            self.line_index += 1
            return "continue"
        return "pass"

    def handle_class(self, template_string, line, match_start, match_class):
        """
        :param template_string:
        :param line:
        :param match_start:
        :param match_class:
        :return:
        """
        if not match_class:  # we face a class
            return "pass"
        self.stack_template.append(template_string)
        self.stack.append('class_now')
        class_name = match_class.group(3)

        # class template specializations: class A<u,Node<u> >
        if '<' in class_name:
            k = line.index('<')
            fit = 1
            for ii in range(k + 1, len(line)):
                if line[ii] == '<':
                    fit += 1
                if line[ii] == '>':
                    fit -= 1
                if fit == 0:
                    break
            class_name += line[k + 1:ii + 1]
        logging.info('class_name[%s]', class_name)
        self.stack_class.append(class_name)
        while not match_start:
            self.line_index += 1
            line = self.input_content[self.line_index]
            match_start = pattern_start.search(line)
        self.line_index += 1
        return "continue"

    def handle_template(self):
        line = self.input_content[self.line_index]
        match_template = pattern_template.search(line)
        template_string = ''
        if match_template:
            match_template_end = pattern_template_end.search(line)
            template_string = line
            while not match_template_end:
                self.line_index += 1
                line = self.input_content[self.line_index]
                template_string += line
                match_template_end = pattern_template_end.search(line)
            self.line_index += 1
        return template_string

    def handle_namespace(self):
        line = self.input_content[self.line_index]
        match_namespace = pattern_namespace.search(line)

        while match_namespace:
            self.output_fd.write(line + '\n')
            self.stack.append('namespace_now')
            self.line_index += 1
            line = self.input_content[self.line_index]
            match_namespace = pattern_namespace.search(line)

    def handle_externc(self):
        line = self.input_content[self.line_index]
        match_namespace = pattern_externc.search(line)
 
        while match_namespace:
            self.output_fd.write(line + '\n')
            self.stack.append('externc_now')
            self.line_index += 1
            line = self.input_content[self.line_index]
            match_namespace = pattern_externc.search(line)          

    def handle_global_var(self):
        line = self.input_content[self.line_index]
        match_gvar = pattern_gvar.search(line)
        while match_gvar:
            content = ""
            if match_gvar.group(1):
                content += match_gvar.group(1)
            content += match_gvar.group(2) + ' ' + match_gvar.group(3)
            if match_gvar.group(4):
                content += match_gvar.group(4)
            content += ';'
            self.output_fd.write(content + '\n')
            self.line_index += 1
            line = self.input_content[self.line_index]
            match_gvar = pattern_gvar.search(line)

    def handle_normal_func(self, line, template_string):
        template_line = ''
        self.stack_template.append(template_string)
        if self.stack_template[-1] != '':
            template_line = re.sub(r'\s*template', 'template', self.stack_template[-1])
            # change '< class T = a, class U = A(3)>' to '<class T, class U>'
            template_line = re.sub(r'\s*=.*>(\s*)$', r'>\1', template_line)
            template_line = re.sub(r'\s*=.*,', ',', template_line)
            template_line = re.sub(r'\s*=.*', '', template_line)
        line = re.sub(r'\s*=.*,', ',', line)
        line = re.sub(r'\s*=.*\)', ')', line)
        line = template_line + line
        self.stack_template.pop()
        func_name = re.search(r'^.*\)', line, re.MULTILINE | re.DOTALL).group()
        logging.info("line[%s]", line)
        logging.info("func_name[%s]", func_name)
        return line, func_name

    def handle_class_member_func(self, line, template_string):
        template_line = ''
        x = ''
        if template_string != '':
            template_string = re.sub(r'\s*template', 'template', template_string)
            template_string = re.sub(r'\s*=.*>(\s*)$', r'>\1', template_string)
            template_string = re.sub(r'\s*=.*,', ',', template_string)
            template_string = re.sub(r'\s*=.*', '', template_string)
        if self.stack_template[-1] != '':
            if not (re.search(r'<\s*>', self.stack_template[-1])):
                template_line = re.sub(r'^\s*template', 'template', self.stack_template[-1])
                if not (re.search(r'<.*>', self.stack_class[-1])):
                    # for x we get like template<class T, typename U> -> <T,U>
                    x = re.sub(r'template\s*<', '<', template_line)  # remove template -> <class T, typename U>
                    x = re.sub(r'\n', '', x)
                    x = re.sub(r'\s*=.*,', ',', x)
                    x = re.sub(r'\s*=.*\>', '>', x)
                    x = x.rstrip()  # remove \n
                    x = re.sub(r'(class|typename)\s+|(<class>|<typename>\s*class)', '',
                               x)  # remove class,typename ->  <T, U>
                    x = re.sub(r'<\s+', '<', x)
                    x = re.sub(r'\s+>', '>', x)
                    x = re.sub(r'\s+,', ',', x)
                    x = re.sub(r',\s+', ', ', x)
        line = re.sub(r'\s*=\s+0', '', line)
        line = re.sub(r'\s*=\s+.*,', ',', line)
        line = re.sub(r'\s*=\s+.*\)', ')', line)
        logging.info("x[%s]\nline[%s]", x, line)
        # if the function is long, void ABC::foo()
        # breaks into two lines void ABC::\n foo()
        rep_fmt = '%s%s::{}%s' % (self.stack_class[-1], x, r'\1(')
        temp_line = pattern_func_name.sub(rep_fmt.format(''), line, count=1)
        if len(temp_line) > MAX_CODE_LEN_PER_LINE:
            line = pattern_func_name.sub(rep_fmt.format('\n'), line, count=1)
        else:
            line = temp_line
        logging.info("line[%s]", line)
        # add template as the above if there is one
        template_line = re.sub(r'\s*=.*>(\s*)$', r'>\1', template_line)
        template_line = re.sub(r'\s*=.*,', ',', template_line)
        template_line = re.sub(r'\s*=.*', '', template_line)
        line = template_line + template_string + line
        func_name = re.search(r'^.*\)', line, re.MULTILINE | re.DOTALL).group()
        logging.info("line[%s]", line)
        logging.info("func_name[%s]", func_name)
        return line, func_name

    def write_func_content(self, content, func_name, need_generate):
        if not (func_name in self.func_list_exist) and need_generate:
            self.output_fd.write(content)
            self.func_list_exist.append(func_name)
            logging.info('add func:[%s]', func_name)

    def gen_comment(self, start_i):
        comment_line = ''
        # Function comments are on top of function declarations, copy them over
        k = start_i - 1  # one line before this func start
        if pattern_template.search(self.input_content[k]):
            k -= 1
        if pattern_comment_2_end.search(self.input_content[k]):
            comment_line = self.input_content[k].lstrip()
            while not pattern_comment_2_start.search(self.input_content[k]):
                k -= 1
                comment_line = self.input_content[k].lstrip() + comment_line
        else:
            for j in range(k, 0, -1):
                c_line = self.input_content[j]
                if pattern_comment.search(c_line):
                    c_line = re.sub(r'\s*//', '//', c_line)
                    comment_line = c_line + comment_line
                else:
                    break
        return comment_line

    @staticmethod
    def get_return_statements(func):
        if func.find("T &aclArray<T>::operator") != -1:
            return "static T var; return var;\n"
        func = pat_format_func.sub(r'\1\3\2', func)
        m = pat_search_func.search(func)
        if not m:
            return None
        logging.info('ret_type: %s, class_name: %s, func_name: %s', *m.group('ret_type', 'class_name', 'func_name'))
        type_cls_func_name = '%s %s::%s' % m.group('ret_type', 'class_name', 'func_name')
        if type_cls_func_name in RETURN_STATEMENTS:
            logging.info('type_cls_func_name:[%s] matched!', type_cls_func_name)
            return RETURN_STATEMENTS[type_cls_func_name]
        type_cls_name = '%s %s::' % m.group('ret_type', 'class_name')
        if type_cls_name in RETURN_STATEMENTS:
            logging.info('type_cls_name:[%s] matched!', type_cls_name)
            return RETURN_STATEMENTS[type_cls_name]
        type_only = m.group('ret_type')
        if type_only in RETURN_STATEMENTS:
            logging.info('type_only:[%s] matched!', type_only)
            return RETURN_STATEMENTS[type_only]
        return None

    @staticmethod
    def implement_function(func):
        if "VISIBILITY_EXPORT" in func or "ACL_FUNC_VISIBILITY" in func:
            func = func.replace("VISIBILITY_EXPORT ", "").replace("ACL_FUNC_VISIBILITY ", "")        
        function_def = ''
        function_def += '{\n'

        return_statements = H2CC.get_return_statements(func)
        if return_statements is not None:
            function_def += return_statements
        else:
            all_items = func.split()
            start = 0
            is_constructor_or_destructor = False
            constructor_idx = 0
            return_type = all_items[start]
            if return_type == "const":
                start += 1
                return_type = all_items[start]
            if return_type.startswith(('std::map', 'std::set', 'std::vector')):
                return_type = "std::map"
            if return_type.endswith('*') or (
                    len(all_items) > start + 1 and (all_items[start + 1].startswith('*') and 
                    not all_items[start + 1].startswith('*&'))) or return_type.startswith(
                'std::unique_ptr'):
                return_type = "Ptr"
            if len(all_items) > start + 1 and all_items[start + 1].startswith('&'):
                return_type += "&"
            if pattern_template.search(all_items[0]) :
                for idx in range(1, len(all_items) - 1):
                    if all_items[idx].endswith('>'):
                        constructor_idx = idx + 1
                        break
                return_type = all_items[constructor_idx]
            if return_type.startswith('std::complex'):
                return_type = "complex"
            if len(all_items) > start + 1 and all_items[start + 1].startswith('*&'):
                return_type = all_items[start] + '*&'
            start_spilts = all_items[constructor_idx].split('::')
            if len(start_spilts) > 1 and start_spilts[1].startswith(start_spilts[0]):
                is_constructor_or_destructor = True

            if RETURN_STATEMENTS.__contains__(return_type) and not is_constructor_or_destructor:
                function_def += RETURN_STATEMENTS[return_type]
            else:
                logging.warning("Unhandled return type[%s]", return_type)

        function_def += '\n'
        function_def += '}\n'
        function_def += '\n'
        return function_def


def collect_header_files(path, shared_includes_content=None, is_head=False):
    """
    :param path:
    :return:
    """
    header_files = []
    for root, dirs, files in os.walk(path):
        files.sort()
        for file in files:
            if file.find("git") >= 0:
                continue
            if not file.endswith('.h'):
                continue
            file_path = os.path.join(root, file)
            file_path = file_path.replace('\\', '/')
            if file_path not in header_files:
                header_files.append(file_path)
            include_str = '#include "{}"\n'.format(file_path[path.rindex('/') + 1:])
            if shared_includes_content is not None and include_str not in shared_includes_content:
                shared_includes_content.append(include_str)
 
    if is_head:
        return shared_includes_content
    return header_files


def generate_stub_file(inc_dir, out_cc_dir, shared_includes_content):
    """
    :param inc_dir:
    :param out_cc_dir:
    :return:
    """
    target_header_files = collect_header_files(inc_dir)
    for header_file in target_header_files:
        if not file_endswith_white_list_suffix(header_file):
            continue
        cc_file = re.sub(r'([^/]+)\.h$', r'stub_\1.cc', header_file)
        h_2_cc = H2CC(header_file, out_cc_dir + cc_file[cc_file.rindex('/') + 1:], shared_includes_content)
        h_2_cc.h2cc()


def gen_code(inc_dir, out_cc_dir):
    """
    :param inc_dir:
    :param out_cc_dir:
    :return:
    """
    if not inc_dir.endswith('/'):
        inc_dir += '/'
    if not out_cc_dir.endswith('/'):
        out_cc_dir += '/'
    shared_includes_files = []
    for include_dir_key_word in include_dir_key_words:
        shared_includes_files = collect_header_files(inc_dir + include_dir_key_word, shared_includes_files, True)

    # for acl error code
    shared_includes_files.append('using namespace std;\n')
    for include_dir_key_word in include_dir_key_words:
        generate_stub_file(inc_dir + include_dir_key_word, out_cc_dir, shared_includes_files)


def main():
    if len(sys.argv) != 3:
        logging.error("script %s must have 2 input parameters!" % sys.argv[0])
        return
    inc_dir = sys.argv[1]
    out_cc_dir = sys.argv[2]
    gen_code(inc_dir, out_cc_dir)


if __name__ == '__main__':
    main()
