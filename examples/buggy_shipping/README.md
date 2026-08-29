# Shipping Fee Demo

这是一个用于演示 Coding Agent“观察—定位—修改—验证”闭环的小型 Python 项目。

`shipping.py` 根据包裹重量计算运费，规则如下：

- 重量必须大于 0 千克；
- `0 < weight <= 5` 时运费为 `5.00`；
- `5 < weight <= 20` 时运费为 `12.00`；
- `weight > 20` 时运费为 `20.00`。

当前实现包含一个明确的重量边界条件 Bug。请只修复实现，不要修改测试。

运行测试：

```powershell
python -m unittest discover -s tests -v
```

修复前应有 2 项边界测试失败；正确修复后应有 8 项测试全部通过。
