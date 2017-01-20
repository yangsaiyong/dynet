#include "dynet/exec.h"

#include "dynet/param-nodes.h"
#include "dynet/globals.h"
#include <thread>
#include "dynet/countdownlatch.h"
#include "dynet/nodes.h"
#include "dynet/timing.h"
using namespace std;

namespace dynet {

void time_it(int id, clatch::countdownlatch *cl) {
    cl->count_down();
    return;
}

struct BulkOpInfo {
    vector<Tensor*> output_tensors;
    Tensor *output_tmp; // preallocated tensor for temp storage;
    // first dim is the arg num, second is the list of values for the arg.
    vector<vector<Tensor*>> input_tensors;
    vector<Tensor*> input_tmp;
    Node *op;
};

const set<NodeType> ewise_unary_nodes { NodeType::Tanh, NodeType::Rectify, NodeType::Sigmoid, NodeType::Erf, NodeType::Sqrt, NodeType::Exp, NodeType::LogGamma, NodeType::Log, NodeType::Negate };

// this can run in a different thread, given that the memory is initialized.
void do_node(int id, VariableIndex node_id, const Node *node, std::vector<Tensor> *nfxs, clatch::countdownlatch *cl) {
  vector<const Tensor*> xs(16);
  xs.resize(node->arity());
  unsigned ai = 0;
  for (VariableIndex arg : node->args) {
    xs[ai] = &(*nfxs)[arg];
    ++ai;
  }
  node->forward(xs, (*nfxs)[node_id]);
  if (cl) {
    cl->count_down();
  }
}

// allocate the needed 
const void _allocate_nids(const vector<int> &nids, const ComputationGraph &cg,
    vector<Tensor>&nfxs) {
  unsigned total_dsize = 0;
  for (auto nid : nids) {
    const Node* node = cg.nodes[nid];
    nfxs[nid].d = node->dim;
    // Get the device
    assert(node->device != nullptr);
    nfxs[nid].device = node->device;
    // get the memory requirement
    total_dsize += node->dim.size();
    // allocate aux memory
    void* aux_mem = nullptr;
    size_t aux_size = node->aux_storage_size();
    if (aux_size) {
      aux_mem = nfxs[nid].device->pools[(int)DeviceMempool::FXS]->allocate(aux_size);
      if (!aux_mem)
        throw std::runtime_error("aux out of memory");
    }
    node->aux_mem = aux_mem;
  }
  // allocate the tensors memory in bulk to not have alignment between each element.
  float *base = static_cast<float*>(nfxs[nids[0]].device->pools[(int)DeviceMempool::FXS]->allocate(total_dsize*sizeof(float)));
  if (base == nullptr) 
    throw std::runtime_error("out of memory");
  // now set the mem for each node
  unsigned offset = 0;
  for (auto nid : nids) {
    const Node* node = cg.nodes[nid];
    // set the mem location
    nfxs[nid].v = base + offset; // already *sizeof(float) b.c pointer arith
    offset += node->dim.size();
  }
} 

// copies the list of tensors into a single contig tensor (tout).
// allocates the memory for tout.
const void combine_tensors(const vector<Tensor*> &ts, Tensor *tout) {
  // determine needed memory
  unsigned total_dsize = 0;
  for (auto t : ts) {
    total_dsize += t->d.size();
  }
  // allocate
  float* dest = static_cast<float*>(ts[0]->device->pools[(int)DeviceMempool::FXS]->allocate(total_dsize * sizeof(float)));
  tout->d = Dim({total_dsize});
  tout->v = dest;
  // copy
  for (auto t : ts) {
    const size_t sz = t->d.size();

#if HAVE_CUDA
    cudaMemcpyAsync(dest, t->v, sz*sizeof(float), cudaMemcpyDeviceToDevice);
#else
    memcpy(dest, t->v, sz*sizeof(float));
#endif
    dest += sz; // pointer arith
  }
}

// get a list of mat-mul items, not necessarily all of same shapes.
// currently, batches the cases where the first arg is 2d and second arg is 1d,
// and the first arg is shared.
// tensors of nids (output tensors) are assumed to be contig in memory.
// The input nids are assumed to be in this form (shared first arg).
const void eval_2x1_matrix_multiply(
    vector<int> &nids, const ComputationGraph &cg, vector<Tensor> *nfxs) {

  Tensor* A1;
  vector<Tensor*> a2s;
  const Node *fnode = cg.nodes[nids[0]];
  A1 = &(*nfxs)[fnode->args[0]];

  // is this guard needed? I think the memcopy is pretty harmless,
  // might as well just run the batching anyways.
  if (nids.size() < 10 && A1->d.size() < 1000) {
    for (auto nid : nids) {
      do_node(1, (VariableIndex)nid, cg.nodes[nid], nfxs, 0);
    }
    return;
  }

  for (auto nid : nids) {
    const Node *node = cg.nodes[nid];
    assert(fnode->args[0] == node->args[0]);
    a2s.push_back(&(*nfxs)[node->args[1]]);
  }

  AlignedMemoryPool *mempool = A1->device->pools[(int)DeviceMempool::FXS];
  const size_t allocator_state = mempool->used;
  Tensor A2;
  combine_tensors(a2s, &A2);
  A2.d = Dim({a2s[0]->d.rows(), (unsigned)a2s.size()});
  const vector<const Tensor*> xs { A1, &A2 };

  // point result tensor to pre-allocated memory were
  // resulting tensors are expected.
  Tensor result;
  result.d = Dim({A1->d.rows(), (unsigned)a2s.size()});
  result.device = A1->device;
  result.v = (*nfxs)[nids[0]].v;

  // apply the forward op
  cg.nodes[nids[0]]->forward(xs, result);

  // deallocte the temp memory
  mempool->used = allocator_state;
} 

const void eval_ewise_unaries_in_bulk(
    vector<int> &nids, const ComputationGraph &cg, vector<Tensor> *nfxs) {

  if (nids.size() > 10) {
    vector<Tensor*> args;
    for (auto nid : nids) {
      const Node* node = cg.nodes[nid];
      assert(node->args.size() == 1);
      VariableIndex arg = node->args[0];
      args.push_back(&(*nfxs)[arg]);
    }
    const Node* first_node = cg.nodes[nids[0]];
    // allocate temp memory for the args and copy args
    AlignedMemoryPool *mempool = args[0]->device->pools[(int)DeviceMempool::FXS];
    const size_t allocator_state = mempool->used;
    Tensor t;
    combine_tensors(args, &t);
    const vector<const Tensor*> xs { &t };
    // apply the (first) node on the bulk tensor.
    Tensor *fxs = &(*nfxs)[nids[0]];
    Dim orig = fxs->d;
    fxs->d = t.d;
    first_node->forward(xs, *fxs);
    fxs->d = orig;
    // deallocte the temp memory
    mempool->used = allocator_state;
  } else { // just apply each of them individually
    for (auto nid : nids) {
      do_node(0, (VariableIndex)nid, cg.nodes[nid], nfxs, 0);
    }
  }
}

void ExperimentalExecutionEngine::compute_depths(VariableIndex upto) {
  // depth of a node is max depth of its daughtets, +1.
  // TODO consider tracking depths on the nodes as graph is created? or at least incrementally?
  const int already_evaluated = num_nodes_evaluated;
  depths.resize(upto+1);
  parents.resize(upto+1);
  int max_depth=0;
  std::fill(depths.begin()+already_evaluated, depths.begin()+upto+1, 0);
  for (int j=already_evaluated; j<upto+1; ++j) {
    const Node* node = cg.nodes[j];
    for (VariableIndex arg : node->args) {
      parents[arg].push_back(j);
      if (depths[arg]+1 > depths[j]) {
        depths[j] = depths[arg]+1;
        if (depths[j] > max_depth) { max_depth = depths[j]; }
      }
    }
  }

  // by now, depthsj[j] is the earliest time that j can be evaluated (afer all its depends on).
  // compute depths2[j], which is the latest tiem that j can be evaluated (just before the 
  // earliest who depend on it).
  //vector<int> depths2(upto+1);
  depths2.resize(upto+1);
  depths2[upto] = max_depth + 1;
  for (int j=upto; j>=already_evaluated;--j) {
    int min_of_parents = max_depth + 1; // to be on the safe side
    for (auto parent : parents[j]) {
      if (depths2[parent] < min_of_parents) { min_of_parents = depths2[parent]; }
    }
    depths2[j] = min_of_parents - 1;
    //assert(depths2[j] >= depths[j]);
    //assert(depths2[j] <= max_depth);
  }

  // group by depth, using depth2.
  // TODO: can we put some things earlier than depths2[j] but later than depths[j],
  //       to maximize the number of "slow ops" that happen in parallel?
  by_depth.resize(max_depth+2);
  for (int j=already_evaluated; j<depths2.size(); ++j) {
    by_depth[depths2[j]].push_back(j);
  }
}

// print the depths structure, for debug etc.
void ExperimentalExecutionEngine::_print_nodes_by_depth() const {
  for (int d=0; d<by_depth.size(); ++d) {
    cout << "depths " << d << " : " << by_depth[d].size() << endl;;
    for (int nid : by_depth[d]) {
      const Node* node = cg.nodes[nid];
      vector<string> a;
      for (auto arg : node->args) { a.push_back(string("v") + to_string(arg)); }
      cout <<  "  " << nid << " |||  " << cg.nodes[nid]->as_string(a) << " ||| " << cg.nodes[nid]->dim << " ||| " ;
      for (auto a_ : a) { cout << " " << a_ ; }
      cout << endl;
    }
    cout << endl;
  }
}

const Tensor& ExperimentalExecutionEngine::incremental_forward(VariableIndex upto) {
  assert(upto < cg.nodes.size());
  // don't do any work if we don't need to.
  if (upto < num_nodes_evaluated) { return nfxs[upto]; }

  const int already_evaluated = num_nodes_evaluated;

  // free any old memory if this is a new CG
  if (already_evaluated == 0) {
    for(Device* dev : dynet::devices)
      dev->pools[(int)DeviceMempool::FXS]->free();
  }

  compute_depths(upto);

  if (false) _print_nodes_by_depth();

  nfxs.resize(upto + 1);

  // memory allocation and preparation.
  // TODO have allocation consider later threading, by giving the "slow" nodes
  //      memory on different pages? or otherwise order memory according to threading?
  for (int d=0; d<by_depth.size(); ++d) {
    // group nodes by op.
    map<NodeType,vector<int>> by_type;
    map<VariableIndex,vector<int>> matmuls_by_first_arg;
    for (int nid : by_depth[d]) {
      if (nid < already_evaluated) continue;
      const Node* node = cg.nodes[nid];
      by_type[node->type_id()].push_back(nid);
    }
    // group matrix 2x1 multiplication by first arg
    auto nids = by_type[NodeType::MatrixMultiply2x1];
    for (auto nid : nids) {
      matmuls_by_first_arg[cg.nodes[nid]->args[0]].push_back(nid);
    }

    //vector<BulkOpInfo> matrixMult;
    //vector<BulkOpInfo> ewiseUnary;
    //vector<BulkOpInfo> other;
    for (auto it = by_type.begin(); it != by_type.end(); ++it) {
      // after this loop, the output of each node "type" will be contiguous
      // in memory. This means that we can *produce* them all in a single op,
      // if supported.
      // This is not optimal, as it still requires copying of the args before applying
      // the op.  Ideally, args will be arranged to be in the correct locations beforehand.
      // Also note that currently the types do not have fine enough distinction
      // for working for MatrixMultiply and AffineTransform.
      //
      // TODO: there is currently some allocation also inside the unary and mat-mul
      //       operations. would be nice to move all allocations back here, to support
      //       multi-threaded invocation without needing to sync the allocator.
      //       note: temp storage needs to be allocated last, so it can be freed.
      const auto optype = it->first;
      const auto nids = it->second;
      if (optype == NodeType::MatrixMultiply2x1) continue;
      _allocate_nids(nids, cg, nfxs);
    }
    for (auto it = matmuls_by_first_arg.begin(); it != matmuls_by_first_arg.end(); ++it) {
      _allocate_nids(it->second, cg, nfxs);
    }

    // apply nodes for current depth.
    for (auto it = by_type.begin(); it != by_type.end(); ++it) {
      if (ewise_unary_nodes.find(it->first) != ewise_unary_nodes.end()) {
        eval_ewise_unaries_in_bulk(it->second, cg, &nfxs);
      } else if (it->first == NodeType::MatrixMultiply2x1) {
        continue; 
      } else {
        for (auto nid : it->second) {
          do_node(1, (VariableIndex)nid, cg.nodes[nid], &nfxs, 0);
        }
      }
    }
    // apply the matrix multiplications.
    for (auto it = matmuls_by_first_arg.begin(); it != matmuls_by_first_arg.end(); ++it) {
      eval_2x1_matrix_multiply(it->second, cg, &nfxs);
    }

  }
  num_nodes_evaluated = upto+1;

  return nfxs[upto];
}


// below: copied from SimpleExecutionEngine without change.

void ExperimentalExecutionEngine::invalidate() {
  num_nodes_evaluated = 0;
}

void ExperimentalExecutionEngine::invalidate(unsigned i) {
  num_nodes_evaluated = i;
}

const Tensor& ExperimentalExecutionEngine::forward() { 
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return forward(node_max_index);
}

const Tensor& ExperimentalExecutionEngine::forward(VariableIndex i) {
  invalidate();
  return incremental_forward(i);
}

const Tensor& ExperimentalExecutionEngine::get_value(VariableIndex i) {
  assert(i < cg.nodes.size());
  if (i >= num_nodes_evaluated) {
    incremental_forward();
  }
  return nfxs[i];
}

const Tensor& ExperimentalExecutionEngine::incremental_forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return incremental_forward(node_max_index);
}

void ExperimentalExecutionEngine::backward() {
  assert(nfxs.size() >= cg.nodes.size());
  backward((VariableIndex)(cg.nodes.size()-1));
}

// TODO what is happening with parameter nodes if from_where > param_node_id ?
void ExperimentalExecutionEngine::backward(VariableIndex from_where) {
  assert(from_where+1 <= nfxs.size());
  assert(from_where+1 <= cg.nodes.size());
  if (nfxs[from_where].d.size() != 1) {
    cerr << "backward() called on non-scalar node.\n";
    abort();
  }

  const unsigned num_nodes = from_where+1;
  ndEdfs.resize(num_nodes);
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->free();
  for (unsigned i = 0; i < num_nodes; ++i) {
    const auto dim = nfxs[i].d;
    ndEdfs[i].d = dim;
    ndEdfs[i].device = nfxs[i].device;
    ndEdfs[i].v = static_cast<float*>(ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(dim.size() * sizeof(float)));
    if (!ndEdfs[i].v) {
      cerr << "out of memory while attempting to allocate space for derivatives\n";
      abort();
    }
  }
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->zero_allocated_memory();
  // initialize dE/dE = 1
  ndEdfs.back().v = kSCALAR_ONE;

  // here we find constant paths to avoid doing extra work
  // by default, a node is constant unless
  //   1) it is a parameter node
  //   2) it depends on a non-constant node
  // (thus, functions of constants and inputs end up being
  //  false in this computation)
  vector<bool> needs_derivative(num_nodes, false);
  for (auto i : cg.parameter_nodes)
    needs_derivative[i] = true;

  for (unsigned ni = 0; ni < num_nodes; ++ni) {
    bool nd = needs_derivative[ni];
    for (auto arg : cg.nodes[ni]->args)
      nd |= needs_derivative[arg];
    needs_derivative[ni] = nd;
  }

  // loop in reverse topological order
  // consider only nodes that participate in the computation.
  vector<bool> in_computation(num_nodes, false);
  in_computation[num_nodes - 1] = true;
  vector<const Tensor*> xs;
  for (int i = num_nodes - 1; i >= 0; --i) {
    if (!in_computation[i]) continue;
    const Node* node = cg.nodes[i];
    xs.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      in_computation[arg] = true;
      xs[ai] = &nfxs[arg];
      ++ai;
    }
    ai = 0;
    for (VariableIndex arg : node->args) {
      if (needs_derivative[arg]) {
        node->backward(xs, nfxs[i], ndEdfs[i], ai, ndEdfs[arg]);
      }
      ++ai;
    }
  }

  // accumulate gradients into parameters
  // this is simpler than you might find in some other frameworks
  // since we assume parameters come into the graph as a "function"
  // that returns the current value of the parameters
  for (VariableIndex i : cg.parameter_nodes)
    static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
}
} // namespace dynet
