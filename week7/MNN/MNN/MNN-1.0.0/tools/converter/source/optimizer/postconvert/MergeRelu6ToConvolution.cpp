//
//  MergeRelu6ToConvolution.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
#include "MergeToConvolution.hpp"

using namespace MNN;

class MergeRelu6ToConvolution : public MergeToConvolution {
public:
    bool merge2Convolution(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const {
        if (inplaceOp->type == MNN::OpType_ReLU6) {
            convolutionOp->main.AsConvolution2D()->common->relu6 = true;
            return true;
        }
        return false;
    }

    bool merge2Convolution3D(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const {
        if (inplaceOp->type == MNN::OpType_ReLU6) {
            convolutionOp->main.AsConvolution3D()->common->relu6 = true;
            return true;
        }
        return false;
    }
};
static PostConverterRegister<MergeRelu6ToConvolution> __l("MergeRelu6ToConvolution");
