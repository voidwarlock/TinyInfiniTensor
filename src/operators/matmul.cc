#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        if (inputs.size() != 2)
        {
            return std::nullopt;
        }
        size_t rankA = inputs[0] -> getRank();
        size_t rankB = inputs[1] -> getRank();
        
        if (rankA < 2 || rankB < 2) {
            return std::nullopt; // 最后两维必须存在
        }

        const Shape &shapeA = inputs[0]->getDims();
        const Shape &shapeB = inputs[1]->getDims();

        int64_t dimA_last = shapeA[rankA - 1];
        int64_t dimA_sec_Last = shapeA[rankA - 2];
        int64_t dimB_last = shapeB[rankB - 1];
        int64_t dimB_sec_Last = shapeB[rankB - 2];

        if (dimA_last == dimB_sec_Last) {
            dimA_sec_Last = dimA_sec_Last;
            dimB_last = dimB_last;
        }
        else if (dimA_last == dimB_last) {
            dimA_sec_Last = dimA_sec_Last;
            dimB_last = dimB_sec_Last;
        }
        else if (dimA_sec_Last == dimB_sec_Last) {
            dimA_sec_Last = dimA_last;
            dimB_last = dimB_last;
        }
        else if (dimA_sec_Last == dimB_last) {
            dimA_sec_Last = dimA_last;
            dimB_last = dimB_sec_Last;
        }
        else
        {
            return std::nullopt;    
        }
        Shape outputShape;    

        size_t maxRank = std::max(rankA, rankB);
        for (size_t i = 0; i < maxRank - 2; ++i) {
            int64_t dimA = i < rankA ? shapeA[rankA - maxRank + i] : 1;
            int64_t dimB = i < rankB ? shapeB[rankB - maxRank + i] : 1;

            if (dimA != dimB && dimA != 1 && dimB != 1) {
                return std::nullopt; // 广播不兼容
            }
            outputShape.push_back(std::max(dimA, dimB));
        }

        outputShape.push_back(dimA_sec_Last);
        outputShape.push_back(dimB_last);

        return {{outputShape}};
    }

} // namespace infini