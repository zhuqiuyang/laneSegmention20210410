# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class RequantizationRange(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsRequantizationRange(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RequantizationRange()
        x.Init(buf, n + offset)
        return x

    # RequantizationRange
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def RequantizationRangeStart(builder): builder.StartObject(0)
def RequantizationRangeEnd(builder): return builder.EndObject()
