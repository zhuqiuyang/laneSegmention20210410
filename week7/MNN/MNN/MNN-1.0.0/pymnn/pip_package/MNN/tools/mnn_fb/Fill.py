# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class Fill(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFill(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Fill()
        x.Init(buf, n + offset)
        return x

    # Fill
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def FillStart(builder): builder.StartObject(0)
def FillEnd(builder): return builder.EndObject()
