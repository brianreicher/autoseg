import random
import time
import numpy as np
from rusty_mws.rusty_segment_mws import *
from rusty_mws.global_mutex import segment
from rusty_mws.extract_seg_from_luts import extract_segmentation
from funlib.persistence import open_ds, graphs, Array
import mwatershed as mws
from tqdm import tqdm 
from funlib.evaluate import rand_voi

def evaluate_weight_biases(adj_bias, lr_bias, rasters, seg_file, seg_ds, sample_name, edges, adj_scores, lr_scores, merge_function, out_dir, fragments_file, fragments_dataset):
    # Call the function that performs the agglomeration step with the given weight biases
    segment((), edges, adj_scores, lr_scores, merge_function, out_dir, adj_bias, lr_bias)
    extract_segmentation(fragments_file, fragments_dataset, sample_name)

    seg: Array = open_ds(filename=seg_file, ds_name=seg_ds)

    seg: np.ndarray = seg.to_ndarray()

    seg: np.ndarray = np.asarray(seg, dtype=np.uint64)
    
    score_dict: dict = rand_voi(rasters, seg, True)
    print([score_dict[f"voi_split"], score_dict["voi_merge"]])
    return np.mean(a=[score_dict[f"voi_split"], score_dict["voi_merge"]])


def crossover(parent1, parent2):
    # Perform crossover by blending the weight biases of the parents
    alpha = random.uniform(0.0, 1.0)  # Blend factor

    adj_bias_parent1, lr_bias_parent1 = parent1[0], parent1[1]
    adj_bias_parent2, lr_bias_parent2 = parent2[0], parent2[1]

    # Blend the weight biases
    adj_bias_child = alpha * adj_bias_parent1 + (1 - alpha) * adj_bias_parent2
    lr_bias_child = alpha * lr_bias_parent1 + (1 - alpha) * lr_bias_parent2

    return adj_bias_child, lr_bias_child


def mutate(individual, mutation_rate=0.1, mutation_strength=0.1):
    # Perform mutation by adding random noise to the weight biases
    adj_bias, lr_bias = individual

    # Mutate the weight biases with a certain probability
    if random.uniform(0.0, 1.0) < mutation_rate:
        # Add random noise to the weight biases
        adj_bias += random.uniform(-mutation_strength, mutation_strength)
        lr_bias += random.uniform(-mutation_strength, mutation_strength)

    return adj_bias, lr_bias


def evo_algo(population_size, num_generations, adj_bias_range, lr_bias_range,
             seg_file="./validation.zarr", seg_ds="pred_seg", rasters_file="../../data/xpress-challenge.zarr",
             fragments_file="./validation.zarr", fragments_dataset="frag_seg",
             rasters_ds="volumes/validation_gt_rasters", sample_name:str="htem39454661040933637", merge_function="mwatershed"):
    # Initialize the population
    population = []
    for _ in range(population_size):
        adj_bias = random.uniform(*adj_bias_range)
        lr_bias = random.uniform(*lr_bias_range)
        population.append((adj_bias, lr_bias))

    # set the rasters array
    frag = open_ds(fragments_file, fragments_dataset)
    rasters = open_ds(rasters_file, rasters_ds)

    print("Loading rasters . . .")
    rasters = rasters.to_ndarray(frag.roi)
    rasters = np.asarray(rasters, np.uint64)
    
    db_host: str = "mongodb://localhost:27017"
    db_name: str = "seg"
    print("Reading graph from DB ", db_name)
    start = time.time()

    graph_provider = graphs.MongoDbGraphProvider(
        db_name,
        db_host,
        mode="r+",
        nodes_collection=f"{sample_name}_nodes",
        meta_collection=f"{sample_name}_meta",
        edges_collection=sample_name + "_edges_" + merge_function,
        position_attribute=["center_z", "center_y", "center_x"],
    )

    print("Got Graph provider")

    fragments = open_ds(fragments_file, fragments_dataset)

    print("Opened fragments")

    roi = fragments.roi

    print("Getting graph for roi %s" % roi)

    graph = graph_provider.get_graph(roi)

    print("Read graph in %.3fs" % (time.time() - start))

    if graph.number_of_nodes == 0:
        print("No nodes found in roi %s" % roi)
        return

    edges: np.ndarray = np.stack(list(graph.edges), axis=0)
    adj_scores: np.ndarray = np.array([graph.edges[tuple(e)]["adj_weight"] for e in edges]).astype(
        np.float32
    )
    lr_scores: np.ndarray = np.array([graph.edges[tuple(e)]["lr_weight"] for e in edges]).astype(
        np.float32
    )

    out_dir: str = os.path.join(fragments_file, "luts_full")
    os.makedirs(out_dir, exist_ok=True)

    # evo loop
    for generation in range(num_generations):
        print("Generation:", generation)

        # Evaluate the fitness of each individual in the population
        fitness_values = []
        for adj_bias, lr_bias in population:
            print("BIASES:", adj_bias, lr_bias)
            fitness = evaluate_weight_biases(adj_bias, lr_bias, rasters, seg_file,
                                              seg_ds, sample_name, edges, adj_scores,
                                                lr_scores, merge_function, out_dir, fragments_file, fragments_dataset)
            fitness_values.append((adj_bias, lr_bias, fitness))


        # Sort individuals by fitness (descending order)
        fitness_values.sort(key=lambda x: x[2], reverse=True)

        # Select parents for the next generation
        parents = fitness_values[:population_size//2]
        parents = [parent[:2] for parent in parents]


        # Create the next generation through crossover and mutation
        offspring = []
        for _ in range(population_size - len(parents)):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = crossover(parent1, parent2) 
            child = mutate(child) 
            offspring.append(child)

        # Combine parents and offspring to form the new population
        population = parents + offspring

        fvals = sorted(fitness_values, key=lambda x: x[2], reverse=True) #[:len(population)//2]

        # Extract the baises from the fitness values
        adj = [x[0] for x in fvals]
        lr = [x[1] for x in fvals]
        score = [x[2] for x in fvals]

        # Save the biases as an npz file
        np.savez(f"./optimal_biases_{generation}.npz", adj=adj, lr=lr, score=score)

    # Return the best weight biases found in the last generation
    best_biases = sorted(fitness_values, key=lambda x: x[2], reverse=True)[:len(population)]
    return best_biases

def grid_search_optim(adj_bias_range:tuple, lr_bias_range:tuple, 
                      sample_name:str="htem4413041148969302336", 
                      merge_function:str="mwatershed",
                      fragments_file:str="./validation.zarr",
                      fragments_dataset:str="frag_seg",):
    
    db_host: str = "mongodb://localhost:27017"
    db_name: str = "seg"
    print("Reading graph from DB ", db_name)
    start = time.time()

    graph_provider = graphs.MongoDbGraphProvider(
        db_name,
        db_host,
        mode="r+",
        nodes_collection=f"{sample_name}_nodes",
        meta_collection=f"{sample_name}_meta",
        edges_collection=sample_name + "_edges_" + merge_function,
        position_attribute=["center_z", "center_y", "center_x"],
    )

    print("Got Graph provider")

    fragments = open_ds(fragments_file, fragments_dataset)

    print("Opened fragments")

    roi = fragments.roi

    print("Getting graph for roi %s" % roi)

    graph = graph_provider.get_graph(roi)

    print("Read graph in %.3fs" % (time.time() - start))

    if graph.number_of_nodes == 0:
        print("No nodes found in roi %s" % roi)
        return

    edges: np.ndarray = np.stack(list(graph.edges), axis=0)
    adj_scores: np.ndarray = np.array([graph.edges[tuple(e)]["adj_weight"] for e in edges]).astype(
        np.float32
    )
    lr_scores: np.ndarray = np.array([graph.edges[tuple(e)]["lr_weight"] for e in edges]).astype(
        np.float32
    )

    scores: list = []
    print("Running grid search . . .")
    index: int = 0
    for a_bias in tqdm(np.arange(adj_bias_range[0], adj_bias_range[1] + 0.1, 0.1)):
        index+=1
        start_time: float = time.time()
        for l_bias in np.arange(lr_bias_range[0], lr_bias_range[1] + 0.1, 0.1):
            n_seg_run: int = get_num_segs(edges, adj_scores, lr_scores, a_bias, l_bias)
            if 6000<n_seg_run<14000:
                scores.append((a_bias, l_bias, n_seg_run))
        np.savez_compressed("./gridsearch_biases.npz", grid=np.array(sorted(scores, key=lambda x: x[2])))
        print(f"Completed {index}th iteration in {time.time()-start_time} sec")
    print("Completed grid search")


def get_num_segs(edges, adj_scores, lr_scores, adj_bias, lr_bias) -> None:

    edges: list[tuple] = [
        (adj + adj_bias, u, v)
        for adj, (u, v) in zip(adj_scores, edges)
        if not np.isnan(adj) and adj is not None
    ] + [
        (lr_adj + lr_bias, u, v)
        for lr_adj, (u, v) in zip(lr_scores, edges)
        if not np.isnan(lr_adj) and lr_adj is not None
    ]
    edges = sorted(
        edges,
        key=lambda edge: abs(edge[0]),
        reverse=True,
    )
    edges = [(bool(aff > 0), u, v) for aff, u, v in edges]
    lut = mws.cluster(edges)
    inputs, outputs = zip(*lut)

    lut = np.array([inputs, outputs])

    return len(np.unique(lut[1]))



if __name__=="__main__":
    population_size = 5
    num_generations = 1
    adj_bias_range = (-5., 5.)
    lr_bias_range = (-5., 5.)

    print("Optimizing . . .")
    best_biases: list = evo_algo(population_size, num_generations, adj_bias_range, lr_bias_range)
    print("Best weight biases:", best_biases)
