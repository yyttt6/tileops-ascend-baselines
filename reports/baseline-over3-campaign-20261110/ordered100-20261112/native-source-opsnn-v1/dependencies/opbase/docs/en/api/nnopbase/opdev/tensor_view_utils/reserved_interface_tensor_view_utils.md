# Reserved APIs

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** APIs

| API| Description|
| --- | --- |
| CanPickViewAsContiguous(std::initializer_list<const aclTensor *> tensorList) | Checks whether a given tensor list is contiguously stored or transposed for contiguous storage.|
| CanPickViewAsContiguous(const aclTensor *tensor) | Checks whether a given tensor is contiguously stored or transposed for contiguous storage.|
| Validate(const aclTensor *tensor) | Checks whether the view shape, view stride, and view offset of a given tensor are valid.|
