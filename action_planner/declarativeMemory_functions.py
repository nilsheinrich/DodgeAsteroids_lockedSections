"""
...
"""


def generate_new_memory_chunk(observation, SoC, action_goal, utility, declarative_memory):
    """
    define chunk dict and store in declarative_memory.
    new memory chunk is already dict in dict and can be updated with existing declarative_memory.
    new_chunk is returned with content of declarative_memory appended and thus pseudo sorted with newest chunk in the
    beginning (first key) in resulting dict (consisting of dicts)

    instead of observation maybe action field stored?
    """
    new_chunk = {f"{len(declarative_memory) + 1}": {"observation": observation,
                                                    "SoC": SoC,
                                                    "action_goal": action_goal,
                                                    "utility": utility}
                 }
    new_chunk |= declarative_memory
    return new_chunk


def retrieve_chunk(current_observation: list, current_SoC: float, declarative_memory: dict):
    """
    The most similar chunk will provide an action goal that can be applied by only relying memory recall
    (without relying on active computation and generating a new action goal).
    The current observation (abstracted) must be completely identical to the one stored in memory.
    The difference in SoC has to be below specific threshold for chunk to be considered similar.
    """
    highest_allowed_SoC_difference = 0.2
    lowest_SoC_difference = None
    most_similar_chunk = None

    # sorting content of declarative memory by utility first and most recent content second
    # high utility reflects increased retrieval probability
    sorted_keys = sorted(declarative_memory, key=lambda x: (declarative_memory[x]['utility']), reverse=True)
    numberCompetingChunks = 3

    # only loop over numberCompetingChunks in sorted declarative memory as only these chunks will compete for retrieval
    for chunk in {k: declarative_memory[k] for k in list(sorted_keys)[:numberCompetingChunks]}:
        # check for similarities of current_instance and current_SoC with stored chunks and retrieve most similar
        if chunk["observation"] == current_observation:
            SoC_difference = abs(chunk["SoC"] - current_SoC)
            if SoC_difference >= highest_allowed_SoC_difference:
                if lowest_SoC_difference is None:
                    lowest_SoC_difference = SoC_difference
                    most_similar_chunk = chunk

                else:
                    if SoC_difference < lowest_SoC_difference:
                        lowest_SoC_difference = SoC_difference
                        most_similar_chunk = chunk

    return most_similar_chunk, declarative_memory


def increase_utility(memory_chunk, utility_boost=0.1):
    """
    utility boost as arbitrary number reflecting increasing benefit in memory retrieval
    after successful implementation of retrieved action goal
    """

    memory_chunk['utility'] += utility_boost
    return memory_chunk


def clear_declarative_memory(declarative_memory):
    """
    mind wipe
    """
    declarative_memory.clear()
    return declarative_memory
