/** Minimum spanning tree -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @Description
 * Computes minimum spanning tree of a graph using Boruvka's algorithm.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

#include "common.h"
#include "component.h"
#include "cuda_launch_config.hpp"
#include "device_functions.h"
#include "gbar.cuh"
#include "graph.h"
#include "kernelconfig.h"

#include "devel.h"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

__global__ void
dinit(Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent,
      unsigned *partners, bool *processinnextiteration,
      unsigned *goaheadnodeofcomponent, unsigned inpid)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (inpid < graph.nnodes) id = inpid;

    if (id < graph.nnodes) {
        eleminwts[id] = MYINFINITY;
        minwtcomponent[id] = MYINFINITY;
        goaheadnodeofcomponent[id] = graph.nnodes;
        partners[id] = id;
        processinnextiteration[id] = false;
    }
}

__global__ void
dfindelemin(Graph graph, ComponentSpace cs, foru *eleminwts,
            foru *minwtcomponent, unsigned *partners,
            bool *processinnextiteration, unsigned *goaheadnodeofcomponent,
            unsigned inpid)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (inpid < graph.nnodes) id = inpid;

    if (id < graph.nnodes) {
        // if I have a cross-component edge,
        // 	find my minimum wt cross-component edge,
        //	inform my boss about this edge e (atomicMin).
        unsigned src = id;
        unsigned srcboss = cs.find(src);
        unsigned dstboss = graph.nnodes;
        foru minwt = MYINFINITY;
        unsigned degree = graph.getOutDegree(src);
        for (unsigned ii = 0; ii < degree; ++ii) {
            foru wt = graph.getWeight(src, ii);
            if (wt < minwt) {
                unsigned dst = graph.getDestination(src, ii);
                unsigned tempdstboss = cs.find(dst);
                if (srcboss != tempdstboss) {	// cross-component edge.
                    minwt = wt;
                    dstboss = tempdstboss;
                }
            }
        }
        eleminwts[id] = minwt;
        partners[id] = dstboss;

        if (minwt < minwtcomponent[srcboss] && srcboss != dstboss) {
            // inform boss.
            foru oldminwt = atomicMin(&minwtcomponent[srcboss], minwt);
        }
    }
}

__global__ void
dfindelemin2(Graph graph, ComponentSpace cs, foru *eleminwts,
             foru *minwtcomponent, unsigned *partners,
             bool *processinnextiteration, unsigned *goaheadnodeofcomponent,
             unsigned inpid)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < graph.nnodes) {
        unsigned src = id;
        unsigned srcboss = cs.find(src);

        if (eleminwts[id] == minwtcomponent[srcboss] && srcboss != partners[id] && partners[id] != graph.nnodes) {
            unsigned degree = graph.getOutDegree(src);
            for (unsigned ii = 0; ii < degree; ++ii) {
                foru wt = graph.getWeight(src, ii);
                if (wt == eleminwts[id]) {
                    unsigned dst = graph.getDestination(src, ii);
                    unsigned tempdstboss = cs.find(dst);
                    if (tempdstboss == partners[id]) {	// cross-component edge.
                        //atomicMin(&goaheadnodeofcomponent[srcboss], id);

                        if (atomicCAS(&goaheadnodeofcomponent[srcboss], graph.nnodes, id) == graph.nnodes) {
                            //printf("%d: adding %d\n", id, eleminwts[id]);
                            //atomicAdd(wt2, eleminwts[id]);
                        }
                    }
                }
            }
        }
    }
}



__global__ void
verify_min_elem(Graph graph, ComponentSpace cs, foru *eleminwts,
                foru *minwtcomponent, unsigned *partners,
                bool *processinnextiteration, unsigned *goaheadnodeofcomponent,
                unsigned inpid)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (inpid < graph.nnodes) id = inpid;

    if (id < graph.nnodes) {
        if (cs.isBoss(id)) {
            if (goaheadnodeofcomponent[id] == graph.nnodes) {
                //printf("h?\n");
                return;
            }

            unsigned minwt_node = goaheadnodeofcomponent[id];

            unsigned degree = graph.getOutDegree(minwt_node);
            foru minwt = minwtcomponent[id];

            if (minwt == MYINFINITY) {
                return;
            }

            bool minwt_found = false;
            //printf("%d: looking at %d def %d minwt %d\n", id, minwt_node, degree, minwt);
            for (unsigned ii = 0; ii < degree; ++ii) {
                foru wt = graph.getWeight(minwt_node, ii);
                //printf("%d: looking at %d edge %d wt %d (%d)\n", id, minwt_node, ii, wt, minwt);

                if (wt == minwt) {
                    minwt_found = true;
                    unsigned dst = graph.getDestination(minwt_node, ii);
                    unsigned tempdstboss = cs.find(dst);
                    if (tempdstboss == partners[minwt_node] && tempdstboss != id) {
                        processinnextiteration[minwt_node] = true;
                        //printf("%d okay!\n", id);
                        return;
                    }
                }
            }

//	      printf("component %d is wrong %d\n", id, minwt_found);
        }
    }
}

__global__ void
elim_dups(Graph graph, ComponentSpace cs, foru *eleminwts,
          foru *minwtcomponent, unsigned *partners,
          bool *processinnextiteration, unsigned *goaheadnodeofcomponent,
          unsigned inpid)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (inpid < graph.nnodes) id = inpid;

    if (id < graph.nnodes) {
        if (processinnextiteration[id]) {
            unsigned srcc = cs.find(id);
            unsigned dstc = partners[id];

            if (minwtcomponent[dstc] == eleminwts[id]) {
                if (id < goaheadnodeofcomponent[dstc]) {
                    processinnextiteration[id] = false;
                    //printf("duplicate!\n");
                }
            }
        }
    }
}

__global__ void
dfindcompmin(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts,
             foru *minwtcomponent, unsigned *partners,
             bool *processinnextiteration, unsigned *goaheadnodeofcomponent,
             unsigned inpid, GlobalBarrier gb, bool *repeat, unsigned *count)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned id, nthreads = blockDim.x * gridDim.x;
    if (inpid < graph.nnodes) id = inpid;

    unsigned up = (graph.nnodes + nthreads - 1) / nthreads * nthreads;
    unsigned srcboss, dstboss;


    for (id = tid; id < up; id += nthreads) {
        if (id < graph.nnodes && processinnextiteration[id]) {
            srcboss = cs.find(id);
            dstboss = cs.find(partners[id]);
        }

        gb.Sync();

        if (id < graph.nnodes && processinnextiteration[id] && srcboss != dstboss) {
            if (cs.unify(srcboss, dstboss)) {
                atomicAdd(mstwt, eleminwts[id]);
                atomicAdd(count, 1);
                processinnextiteration[id] = false;
                // mark end of processing to avoid getting repeated.
                eleminwts[id] = MYINFINITY;
            } else {
                *repeat = true;
            }
        }

        gb.Sync();
    }
}

int main(int argc, char *argv[])
{
    unsigned *mstwt, hmstwt = 0;
    int iteration = 0;
    Graph hgraph, graph;
    KernelConfig kconf;

    unsigned *partners;
    foru *eleminwts, *minwtcomponent;
    bool *processinnextiteration;
    unsigned *goaheadnodeofcomponent;
    const int nSM = kconf.getNumberOfSMs();

    double starttime, endtime;
    GlobalBarrierLifetime gb;
    const size_t compmintwo_res = maximum_residency(dfindcompmin, 384, 0);
    gb.Setup(nSM * compmintwo_res);

    if (argc != 2) {
        printf("Usage: %s <graph>\n", argv[0]);
        exit(1);
    }

    hgraph.read(argv[1]);

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    hgraph.cudaCopy(graph);

    kconf.setProblemSize(graph.nnodes);
    ComponentSpace cs(graph.nnodes);

    if (cudaMalloc((void **)&mstwt, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating mstwt failed");
    cudaMemcpy(mstwt, &hmstwt, sizeof(hmstwt), cudaMemcpyHostToDevice);	// mstwt = 0.

    if (cudaMalloc((void **)&eleminwts, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating eleminwts failed");
    if (cudaMalloc((void **)&minwtcomponent, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating minwtcomponent failed");
    if (cudaMalloc((void **)&partners, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating partners failed");
    if (cudaMalloc((void **)&processinnextiteration, graph.nnodes * sizeof(bool)) != cudaSuccess) CudaTest("allocating processinnextiteration failed");
    if (cudaMalloc((void **)&goaheadnodeofcomponent, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating goaheadnodeofcomponent failed");

    unsigned prevncomponents, currncomponents = graph.nnodes;

    bool repeat = false, *grepeat;
    cudaMalloc(&grepeat, sizeof(bool) * 1);
    cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice);

    unsigned edgecount = 0, *gedgecount;
    cudaMalloc(&gedgecount, sizeof(unsigned) * 1);
    cudaMemcpy(gedgecount, &edgecount, sizeof(unsigned) * 1, cudaMemcpyHostToDevice);

    printf("finding mst.\n");
    starttime = rtclock();

    do {
        ++iteration;
        prevncomponents = currncomponents;
        dinit <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
        //printf("0 %d\n", cs.numberOfComponentsHost());
        CudaTest("dinit failed");
        dfindelemin <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
        dfindelemin2 <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
        verify_min_elem <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);

        int iter = 1;
        do {
            repeat = false;

            cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice);
            dfindcompmin <<<nSM * compmintwo_res, 384>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent, graph.nnodes, gb, grepeat, gedgecount);
            CudaTest("dfindcompmintwo failed");

            cudaMemcpy(&repeat, grepeat, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
            printf("\t\tinner iter %d\n", iter++);
        } while (repeat); // only required for quicker convergence?

        currncomponents = cs.numberOfComponentsHost();
        cudaMemcpy(&hmstwt, mstwt, sizeof(hmstwt), cudaMemcpyDeviceToHost);
        cudaMemcpy(&edgecount, gedgecount, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost);
        printf("\titeration %d, number of components = %d (%d), mstwt = %u mstedges = %u\n", iteration, currncomponents, prevncomponents, hmstwt, edgecount);
    } while (currncomponents != prevncomponents);
    cudaThreadSynchronize();
    endtime = rtclock();

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    printf("\tmstwt = %u, iterations = %d.\n", hmstwt, iteration);
    printf("\t%s result: weight: %u, components: %u, edges: %u\n", argv[1], hmstwt, currncomponents, edgecount);
    printf("\truntime [mst] = %f ms.\n", 1000 * (endtime - starttime));

    // Cleanup
    cudaFree(eleminwts);
    cudaFree(minwtcomponent);
    cudaFree(processinnextiteration);
    cudaFree(goaheadnodeofcomponent);
    cudaFree(mstwt);
    cudaFree(grepeat);
    cudaFree(gedgecount);

    return 0;
}
