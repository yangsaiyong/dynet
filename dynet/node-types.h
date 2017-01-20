#ifndef DYNET_NODETYPES_H_
#define DYNET_NODETYPES_H_

// TODO what is Erf ?
namespace dynet {
    enum class NodeType { 
        UNK, 
        // elementwise unary
        Tanh, Sigmoid, Rectify, Sqrt, Erf, Exp, LogGamma, Log, Negate,
        // matrix multiplication
        MatrixMultiply2x1,

        // Others to consider later
        MatrixMultiply1x2, MatrixMultiply2x2, MatrixMultiply1x1,
        AffineTransform,
        CwiseMultiply, CwiseQuotient, Concatenate, 
    };
}
#endif
