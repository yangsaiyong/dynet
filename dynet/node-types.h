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

        // Binaries
        CwiseMultiply, CwiseQuotient, BinarySum,
        Pow, Min, Max, DotProduct,

        // Others to consider later
        MatrixMultiply1x2, MatrixMultiply2x2, MatrixMultiply1x1,
        AffineTransform,
        Concatenate, 
    };
}
#endif
