# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class BatchMatMulParam(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsBatchMatMulParam(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BatchMatMulParam()
        x.Init(buf, n + offset)
        return x

    # BatchMatMulParam
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BatchMatMulParam
    def AdjX(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # BatchMatMulParam
    def AdjY(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def BatchMatMulParamStart(builder): builder.StartObject(2)
def BatchMatMulParamAddAdjX(builder, adjX): builder.PrependBoolSlot(0, adjX, 0)
def BatchMatMulParamAddAdjY(builder, adjY): builder.PrependBoolSlot(1, adjY, 0)
def BatchMatMulParamEnd(builder): return builder.EndObject()
