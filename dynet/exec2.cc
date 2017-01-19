#include "dynet/exec.h"

#include "dynet/param-nodes.h"
#include "dynet/globals.h"
#include <thread>
#include <chrono>
#include "dynet/countdownlatch.h"
#include "dynet/nodes.h"
using namespace std;

namespace dynet {

void time_it(int id, clatch::countdownlatch *cl) {
    cl->count_down();
    return;
}

const set<NodeType> ewise_unary_nodes { NodeType::Tanh, NodeType::Rectify, NodeType::Sigmoid, NodeType::Erf, NodeType::Sqrt, NodeType::Exp, NodeType::LogGamma, NodeType::Log, NodeType::Negate };

// this can run in a different thread, given that the memory is initialized.
void do_node(int id, VariableIndex node_id, const Node *node, std::vector<Tensor> *nfxs,
        clatch::countdownlatch *cl) {
    vector<const Tensor*> xs(16);
    xs.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
        xs[ai] = &(*nfxs)[arg];
        ++ai;
    }
    //if (node->slow()) {
        //auto start = std::chrono::system_clock::now(); 
        node->forward(xs, (*nfxs)[node_id]);
        //auto end = std::chrono::system_clock::now(); 
        //std::chrono::duration<float> elapsed_seconds = end-start;
        //cout << "node:" << node_id << " time:" << elapsed_seconds.count() << endl;
    //} else {
    //    node->forward(xs, (*nfxs)[node_id]);
    //}
    if (cl) {
        cl->count_down();
    }
}

const void eval_ewise_unaries_in_bulk(
        vector<int> &nids, const ComputationGraph &cg, vector<Tensor> *nfxs) {

    if (nids.size() > 10) {
        const Node* first_node = cg.nodes[nids[0]];
        size_t size = 0;
        // allocate temp memory for the args and copy args
        AlignedMemoryPool *mempool = (*nfxs)[nids[0]].device->pools[(int)DeviceMempool::FXS];
        const size_t allocator_state = mempool->used;
        float *v_dest = nullptr;
        for (auto nid : nids) {
            const Node* node = cg.nodes[nid];
            assert(node->args.size() == 1);
            VariableIndex arg = node->args[0];
            const size_t sz = cg.nodes[arg]->dim.size() * sizeof(float);
            float *dest = static_cast<float*>(mempool->allocate(sz));
            if (!dest) throw std::runtime_error("aux out of memory");
#if HAVE_CUDA
            cudaMemcpyAsync(dest, (*nfxs)[arg].v, sz, cudaMemcpyDeviceToDevice);
#else
            memcpy(dest, (*nfxs)[arg].v, sz);
#endif
            size += sz;
            if (!v_dest) v_dest = dest;
        }
        // shouldn't I use this and not the sum of sizes bc the alignment issues (is this legit?)
        size = mempool->used - allocator_state; 
        unsigned int new_dim = size/sizeof(float);

        // create a tensor for the bulk args
        Tensor z(Dim({new_dim}), v_dest, (*nfxs)[0].device, (*nfxs)[0].mem_pool);
        const vector<const Tensor*> xs { &z };
        // apply the (first) node on the bulk tensor.
        Tensor *fxs = &(*nfxs)[nids[0]];
        Dim orig = fxs->d;
        fxs->d = Dim({(new_dim)});
        first_node->forward(xs, *fxs);
        fxs->d = orig;
        // deallocte the temp memory
        mempool->used = allocator_state;
    } else { // just apply each of them individually
    //auto start = std::chrono::system_clock::now(); 
        for (auto nid : nids) {
            do_node(0, (VariableIndex)nid, cg.nodes[nid], nfxs, 0);
        }
    //auto end = std::chrono::system_clock::now(); 
    //std::chrono::duration<float> elapsed_seconds = end-start;
    //cout << "alloc and copy :" << elapsed_seconds.count() << endl;
    }
}

const Tensor& ExperimentalExecutionEngine::incremental_forward(VariableIndex upto) {
  assert(upto < cg.nodes.size());
  if (upto < num_nodes_evaluated) { return nfxs[upto]; }

  int already_evaluated = num_nodes_evaluated;
  /*
  for (int i = 0; i < 100; i++) {
    clatch::countdownlatch cl(1);
    auto start = std::chrono::system_clock::now(); 
    //std::thread th(time_it, 1, &cl);
    pool.push(time_it, &cl);
    cl.await();
    //th.join();
    auto end = std::chrono::system_clock::now(); 
    std::chrono::duration<float> elapsed_seconds = end-start;
    cout << elapsed_seconds.count() << endl;
  }
  exit(1);
  */

  // depth of a node is max depth of its daughtets, +1.
  // TODO consider tracking depths on the nodes as graph is created? or at least incrementally?
  depths.resize(upto+1);
  parents.resize(upto+1);
  int max_depth=0;
  for (int j=already_evaluated; j<upto+1; ++j) { depths[j] = 0; } // how do I initialize to zeros?
  for (int j=already_evaluated; j<upto+1; ++j) {
      const Node* node = cg.nodes[j];
      for (VariableIndex arg : node->args) {
          parents[arg].push_back(j); // track parents
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
  
  // print the depths structure
  if (false) {
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

  // free any old memory if this is a new CG
  if (num_nodes_evaluated == 0)
    for(Device* dev : dynet::devices)
      dev->pools[(int)DeviceMempool::FXS]->free();

    nfxs.resize(upto + 1);

    // memory allocation and preparation.
    // TODO have allocation consider later threading, by giving the "slow" nodes
    //      memory on different pages? or otherwise order memory according to threading?
    for (int d=0; d<by_depth.size(); ++d) {
        map<NodeType,vector<int>> by_type;
        for (int nid : by_depth[d]) {
            if (nid < already_evaluated) continue;
            const Node* node = cg.nodes[nid];
            by_type[node->type_id()].push_back(nid);
        }
        for (auto it = by_type.begin(); it != by_type.end(); ++it) {
            // after this loop, the output of each node "type" will be contiguous
            // in memory. This means that we can *produce* them all in a single op,
            // if supported.
            // This is not optimal, as it still requires copying of the args before applying
            // the op.  Ideally, args will be arranged to be in the correct locations beforehand.
            // Also note that currently the types do not have fine enough distinction
            // for working for MatrixMultiply and AffineTransform.
            for (auto nid : it->second) {
                const Node* node = cg.nodes[nid];
                nfxs[nid].d = node->dim;
                // Get the device
                assert(node->device != nullptr);
                nfxs[nid].device = node->device;

                // Get the memory
                nfxs[nid].v = static_cast<float*>(nfxs[nid].device->pools[(int)DeviceMempool::FXS]->allocate(node->dim.size() * sizeof(float)));
                /*
                   cout << "allocating memory for " << num_nodes_evaluated << " size: " << node->dim.size() * sizeof(float) << " addr: " << nfxs[num_nodes_evaluated].v;
                   for (int i = 0; i < 5; i++) {
                   cout << " " << nfxs[num_nodes_evaluated].v[i];
                   }
                   cout << endl; 
                 */
                if (nfxs[nid].v == nullptr)
                    throw std::runtime_error("out of memory");
                void* aux_mem = nullptr;
                size_t aux_size = node->aux_storage_size();
                if (aux_size) {
                    aux_mem = nfxs[nid].device->pools[(int)DeviceMempool::FXS]->allocate(aux_size);
                    if (!aux_mem)
                        throw std::runtime_error("aux out of memory");
                }
                node->aux_mem = aux_mem;
            }
        }
        // apply nodes for current depth
        for (auto it = by_type.begin(); it != by_type.end(); ++it) {
            if (ewise_unary_nodes.find(it->first) != ewise_unary_nodes.end()) {
                eval_ewise_unaries_in_bulk(it->second, cg, &nfxs);
            } else {
                for (auto nid : it->second) {
                    do_node(1, (VariableIndex)nid, cg.nodes[nid], &nfxs, 0);
                }
            }
        }

    }
    num_nodes_evaluated = upto; // or is it upto + 1?

    return nfxs[upto];
    /*
        for (unsigned nid = already_evaluated; nid <= upto; ++nid) {
        const Node* node = cg.nodes[nid];
        do_node(1, (VariableIndex)nid, node, &nfxs, 0);
        }*/

    // TODO this is a mess now
    // memory is allocated, execute graph (possibly in threads)
    for (int d=0; d<by_depth.size(); ++d) {
        map<NodeType,vector<int>> by_type;
        for (int nid : by_depth[d]) {
            if (nid < already_evaluated) continue;
            const Node* node = cg.nodes[nid];
            by_type[node->type_id()].push_back(nid);
        }
        for (auto it = by_type.begin(); it != by_type.end(); ++it) {
            vector<int> slow_ops;
            int first_slow_op = -1;
            for (int nid : by_depth[d]) {
                if (nid < already_evaluated) continue;
                if (cg.nodes[nid]->slow()) slow_ops.push_back(nid);
            }
            if (slow_ops.size() > 0) {
                first_slow_op = slow_ops.back();
                slow_ops.pop_back();
            }
            int n_thread_ops = slow_ops.size();
            //if (slows < 2) slows = 0;
            //cout << slow_ops.size() << "/" << by_depth[d].size() << endl;
            if (ncpu <= 1) n_thread_ops = 0;
            //slows = 0;
            clatch::countdownlatch cl(n_thread_ops);
            // slow nodes in threads
            for (int nid : slow_ops) {
                const Node* node = cg.nodes[nid];
                if (n_thread_ops > 0)
                    pool.push(do_node, (VariableIndex)nid, node, &nfxs, &cl);
                else
                    do_node(1, (VariableIndex)nid, node, &nfxs, 0);
            }
            // first slow op runs in the main thread (concurrently with other slow ops).
            if (first_slow_op > -1) {
                const Node* node = cg.nodes[first_slow_op];
                do_node(1, (VariableIndex)first_slow_op, node, &nfxs, 0);
            }
            // non-slow nodes in main thread (concurrently with the other slow ops).
            for (int nid : by_depth[d]) {
                if (nid < already_evaluated) continue;
                const Node* node = cg.nodes[nid];
                if (!node->slow()) {
                    do_node(1, (VariableIndex)nid, node, &nfxs, 0);
                }
            }
            if (n_thread_ops > 0) { // if needed, wait for the threads to finish.
                //auto start = std::chrono::system_clock::now(); 
                cl.await();
                //auto end = std::chrono::system_clock::now(); 
                //std::chrono::duration<float> elapsed_seconds = end-start;
                //cout << elapsed_seconds.count() << endl;
            }
        }
    }
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
