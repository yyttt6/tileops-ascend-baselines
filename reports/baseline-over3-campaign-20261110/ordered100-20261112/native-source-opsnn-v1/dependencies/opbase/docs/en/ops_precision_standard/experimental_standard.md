# Accuracy Evaluation Standard for Open-Source Ecosystem Operators

## Error Metrics

This standard is used to evaluate accuracy of computational operators contributed to the ecosystem (contributions in the `experimental` directory). Meeting this standard is a necessary condition for ecosystem contributions.
This standard uses the mean relative error and maximum relative error as metrics for evaluation. $actual$ is the actual output result of an NPU, and $golden$ is the golden value.

1. Mean Relative Error (MERE): average relative error across sampling points
   
   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$
   
   When calculating relative error, an epsilon (1e-7) is introduced to avoid the risk of division by zero when the golden value is **0**.
2. Maximum Relative Error (MARE): maximum relative error across sampling points
   
   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

## Evaluation Criteria

**Single benchmark comparison**: direct comparison against a higher-precision reference implementation (a CPU implementation or an Ascend small-operator assembly)

<table style="width: 120%; border-collapse: collapse;">
    <colgroup>
      <col style="width: 25%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
    </colgroup>
    <thead>
      <tr>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>Data Type</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT16</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>BFLOAT16</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT32</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>HiFLOAT32</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT8 E4M3</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT8 E5M2</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong><br>Threshold</strong></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-10</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-7</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-13</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-11</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-3</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-2</sup></td>
      </tr>
    </tbody>
  </table>

**Pass criteria**
A test is passed if MERE < Threshold and MARE < 10 x Threshold.
