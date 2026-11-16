#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "graph/operator_factory.h"
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"

namespace ge {

OpDesc::OpDesc() {}

OpDesc::~OpDesc() {}

Operator OperatorFactory::CreateOperator(char const* a, char const* b) { return Operator(); }

OpDescPtr OpDescUtils::GetOpDescFromOperator(const Operator& oprt)
{
    OpDescPtr ptr = std::make_shared<OpDesc>();
    return ptr;
}

const std::vector<std::pair<std::string, IrInputType>>& OpDesc::GetIrInputs() const
{
    static std::vector<std::pair<std::string, IrInputType>> mp;
    return mp;
}

size_t OpDesc::GetAllInputsSize() const { return 0; }

const std::vector<std::pair<std::string, IrOutputType>>& OpDesc::GetIrOutputs() const
{
    static std::vector<std::pair<std::string, IrOutputType>> mp;
    return mp;
}

size_t OpDesc::GetOutputsSize() const { return 0; }

const std::vector<std::string>& OpDesc::GetIrAttrNames() const
{
    static std::vector<std::string> vec;
    return vec;
}

} // namespace ge