#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize() {
        auto it = ops.begin();
        while (it != ops.end())
        {
            if ((*it)->getOpType() == OpType(OpType::Transpose))
            {
                TransposeObj *transposeOp = dynamic_cast<TransposeObj*>(it->get());
                const std::vector<int> &permutation = transposeOp->getPermute();
                size_t rank = permutation.size();

                auto nextIt = std::next(it);
                if (nextIt != ops.end() && (*nextIt)->getOpType() == OpType(OpType::Transpose))
                {
                    TransposeObj *nextTransposeOp = dynamic_cast<TransposeObj*>(nextIt->get());
                    const std::vector<int> &nextPermutation = nextTransposeOp->getPermute();
                    
                    if (rank == nextPermutation.size() &&
                        permutation[rank - 1] == static_cast<int>(rank - 2) &&
                        permutation[rank - 2] == static_cast<int>(rank - 1) &&
                        nextPermutation[rank - 1] == static_cast<int>(rank - 2) &&
                        nextPermutation[rank - 2] == static_cast<int>(rank - 1))
                    {
                        for (auto &input : transposeOp->getInputs())
                        {
                            if (input)
                            {
                                input->removeTarget(*it);
                            }
                        }
                        for (auto &input : nextTransposeOp->getInputs())
                        {
                            if (input)
                            {
                                input->removeTarget(*nextIt);
                            }
                        }
                        for (auto &output : nextTransposeOp->getOutputs())
                        {
                            if (output)
                            {
                                output->setSource(nullptr);
                                for (auto &target : output->getTargets())
                                {
                                    target->removePredecessors(*nextIt);
                                    target->replaceInput(output, transposeOp->getInputs()[0]);
                                }
                            }
                        }
                        for (auto &output : transposeOp->getOutputs())
                        {
                            if (output)
                            {
                                output->setSource(nullptr);
                                for (auto &target : output->getTargets())
                                {
                                    target->removePredecessors(*it);
                                    target->replaceInput(output, transposeOp->getInputs()[0]);
                                }
                            }
                        }


                        // 移除两个 Transpose 算子的前驱和后继操作
                        for (auto &predecessor : transposeOp->getPredecessors())
                        {
                            predecessor->removeSuccessors(*it);
                        }
                        for (auto &predecessor : nextTransposeOp->getPredecessors())
                        {
                            predecessor->removeSuccessors(*nextIt);
                        }

                        for (auto &successor : transposeOp->getSuccessors())
                        {
                            successor->removePredecessors(*it);
                        }
                        for (auto &successor : nextTransposeOp->getSuccessors())
                        {
                            successor->removePredecessors(*nextIt);
                        }
                        // 从 ops 列表中移除这两个 Transpose 算子
                        removeTensor(nextTransposeOp->getOutputs()[0]);
                        removeTensor(transposeOp->getOutputs()[0]);
                        removeOperator(*nextIt);
                        removeOperator(*it);
                        continue;
                    }
                }
            }
            if ((*it)->getOpType() == OpType(OpType::MatMul)) {
                MatmulObj *matmulOp = dynamic_cast<MatmulObj*>(it->get());

                // 获取 MatMul 的输入张量
                Tensor inputA = (*it)->getInputs()[0];
                Tensor inputB = (*it)->getInputs()[1];
    
                // 检查 inputA 是否来自 Transpose
                Operator sourceA = inputA->getSource();
                if (sourceA && sourceA->getOpType() == OpType(OpType::Transpose) && matmulOp->getTransA()==false) {
                    TransposeObj *transposeOp = dynamic_cast<TransposeObj*>(sourceA.get());
                    const std::vector<int> &permutationA = transposeOp->getPermute();
                    size_t rank = permutationA.size();
                    if (rank >= 2 && permutationA[rank - 1] == static_cast<int>(rank - 2) && permutationA[rank - 2] == static_cast<int>(rank - 1))
                    {
                        matmulOp->setTransA(true);
                        // 移除 Transpose 算子的输入张量的连接
                        for (auto &input : transposeOp->getInputs())
                        {
                            if (input)
                            {
                                input->removeTarget(sourceA);
                            }
                        }

                        // 移除 Transpose 算子的输出张量的连接
                        for (auto &output : transposeOp->getOutputs())
                        {
                            if (output)
                            {
                                output->setSource(nullptr);
                                for (auto &target : output->getTargets())
                                {
                                    target->removePredecessors(sourceA);
                                    target->replaceInput(output, transposeOp->getInputs()[0]);
                                }
                            }
                        }

                        // 移除 Transpose 算子的前驱和后继操作
                        for (auto &predecessor : transposeOp->getPredecessors())
                        {
                            predecessor->removeSuccessors(sourceA);
                        }
                        for (auto &successor : transposeOp->getSuccessors())
                        {
                            successor->removePredecessors(sourceA);
                        }

                        // 从 ops 列表中移除 Transpose 算子
                        removeTensor(transposeOp->getOutputs()[0]);
                        removeOperator(sourceA);
                        continue;

                    }
                }
                // 检查 inputB 是否来自 Transpose
                Operator sourceB = inputB->getSource();
                if (sourceB && sourceB->getOpType() == OpType(OpType::Transpose) && matmulOp->getTransB() == false)
                {
                    TransposeObj *transposeOp = dynamic_cast<TransposeObj*>(sourceB.get());
                    const std::vector<int> &permutationB = transposeOp->getPermute();
                    size_t rank = permutationB.size();
                    if (rank >= 2 && permutationB[rank - 1] == static_cast<int>(rank - 2) && permutationB[rank - 2] == static_cast<int>(rank - 1))
                    {
                        matmulOp->setTransB(true);
                        // 移除 Transpose 算子的输入张量的连接
                        for (auto &input : transposeOp->getInputs())
                        {
                            if (input)
                            {
                                input->removeTarget(sourceB);
                            }
                        }

                        // 移除 Transpose 算子的输出张量的连接
                        for (auto &output : transposeOp->getOutputs())
                        {
                            if (output)
                            {
                                output->setSource(nullptr);
                                for (auto &target : output->getTargets())
                                {
                                    target->removePredecessors(sourceB);
                                    target->replaceInput(output, transposeOp->getInputs()[0]);
                                }
                            }
                        }

                        // 移除 Transpose 算子的前驱和后继操作
                        for (auto &predecessor : transposeOp->getPredecessors())
                        {
                            predecessor->removeSuccessors(sourceB);
                        }
                        for (auto &successor : transposeOp->getSuccessors())
                        {
                            successor->removePredecessors(sourceB);
                        }
                        removeTensor(transposeOp->getOutputs()[0]);
                        removeOperator(sourceB);
                        continue;
                    }
                }
            }
            ++it;
        }
        topo_sort();
    }
    

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        for (auto &tensor : tensors)
        {
            // 计算张量所需的内存大小
            size_t size = tensor->getBytes();
            if (size == 0) {
                continue;
            }
            allocator.alloc(size);
            void* basePtr = allocator.getPtr();
            
            Blob blob = make_ref<BlobObj>(runtime, basePtr);
            tensor->setDataBlob(blob);
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini