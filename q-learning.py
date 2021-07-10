%%writefile qlearning.py

import random
from collections import defaultdict

prev_action = None


from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate

def my_snake(obs):
    return obs.geese[obs.index]

def my_head(obs):
    return my_snake(obs)[0]

def translate_my_head(config, obs, action):
    return translate(position=my_head(obs), direction=action, columns=config.columns, rows=config.rows)

def step(obs):
    # obs.step starts counting at 0; "step number" (for the purpose of hunger and episodeSteps) seems to start counting at 1.
    return obs.step+1 

def action_eats_food(config, obs, action):
    return (translate_my_head(config, obs, action) in obs.food)

# Will my length change due to this action?
def delta_length(config, obs, action):
    l = 0
    if action_eats_food(config, obs, action):
        # I would eat food if I took this action
        l += 1
        
    if step(obs)%config.hunger_rate == 0:
        # I would lose length due to hunger this turn
        l -= 1
    return l

# If I didn't eat starting now, how long would I last before I die of hunger?
def hunger_ttl(config, obs, action):
    snake_len = len(my_snake(obs))
    snake_len += delta_length(config, obs, action) # what WILL my length be after this action?
    
    last_hunger_point = (step(obs)//config.hunger_rate)*config.hunger_rate
    time_die_of_hunger = last_hunger_point+snake_len*config.hunger_rate
    
    return min(config['episodeSteps'], time_die_of_hunger)-step(obs) # or will I not die of hunger before the game is over?


# flood-fill distance to nearest available food 
# (including possibility of distance=0, meaning "you would eat food next turn if you took this action")
def nearest_food_dist(config, obs, action):
    all_goose_loc = set([
        position
        for goose in obs.geese
        for position in goose
    ])
    food_loc = set(obs.food)
    max_dist = config.columns*config.rows
    
    next_head = translate_my_head(config, obs, action)
    '''if next_head in all_goose_loc or action.opposite()==prev_action:
        return max_dist'''
    
    processed_cells = set()
    to_process = [(next_head, 0)]
    while len(to_process) > 0:
        loc, dist = to_process.pop(0)
        if loc not in processed_cells:
            processed_cells.add(loc)
            if loc in all_goose_loc:    
                # going here would crash the goose and (probably) not actually eat any food present.
                # ignore this location and keep searching.
                continue
            if loc in food_loc:
                # Food here! return the distance
                return dist
            else:
                # no food here - where can we go from here?
                next_dist = dist+1
                for next_a in Action:
                    next_loc = translate(loc, next_a, columns=config.columns, rows=config.rows)
                    #if next_loc not in all_goose_loc:
                    to_process.append((next_loc, next_dist))
    
    # ran out of potential cells to process and did not find accessible food - return dummy value
    return max_dist


# How much space can I reach *before* any other goose does?
# counting space which will definitely clear out by the time I get there
# Note: there's definitely some wonkiness in the case of a possible head-on collision (overestimates uncontested space)
def uncontested_space(config, obs, action):
    
    # Enumerate all spaces taken up by geese, and when they will clear out
    goose_parts = {}
    for goose in obs.geese:
        gl = len(goose)
        for i, position in enumerate(goose):
            tt_leave = gl-i # ranges from 1 (tail) to goose length (head)
            # avoid following tail directly, in case the goose eats food (?)
            goose_parts[position] = tt_leave
    
    # If I would crash by taking this action, I have 0 space.
    next_head = translate_my_head(config, obs, action)
    if (next_head in goose_parts and goose_parts[next_head] > 0) or action.opposite()==prev_action:
        return 0
    
    # flood-fill from all geese at once; but keeping my goose separate, going last
    #(because it's actually ahead after taking the action in question)
    
    # track (location, time to get there) tuples for valid locations for a goose to go
    other_to_process = [(g[0], 0) for i,g in enumerate(obs.geese) if (i != obs.index and len(g)>0)]
    me_to_process = [(next_head, 1)]
    me_uncontested = set([next_head])
    
    # spaces which are already 'claimed' - not uncontested
    claimed = set([pos for pos,dist in other_to_process])
    claimed.add(next_head)
    
    other_next_step = []
    me_next_step = []
    
    while len(me_to_process) > 0: # we care only about fully flood-filling *my* space
        # other geese take next step(s)
        for other_loc, other_step in other_to_process:
            for a in Action:
                next_loc = translate(other_loc, a, columns=config.columns, rows=config.rows)
                # can it go there uncontessted?
                if (next_loc not in claimed) and ((next_loc not in goose_parts) or (goose_parts[next_loc] <= other_step)):
                    claimed.add(next_loc)
                    other_next_step.append( (next_loc, other_step+1) )
                    
        # my goose takes next step(s)
        for my_loc, my_step in me_to_process:
            for a in Action:
                next_loc = translate(my_loc, a, columns=config.columns, rows=config.rows)
                if (next_loc not in claimed) and ((next_loc not in goose_parts) or (goose_parts[next_loc] <= my_step)):
                    claimed.add(next_loc)
                    me_next_step.append( (next_loc, my_step+1) )
                    me_uncontested.add(next_loc)
        
        # swap in new to_process lists
        other_to_process = other_next_step
        me_to_process = me_next_step
        other_next_step=[]
        me_next_step=[]
        
    return len(me_uncontested)


# What's my chance of colliding with someone next turn (assuming geese move randomly to a not-currently-occupied spot)?
# factors in both unavoidable collisions(chance=1) and head-on collisions which depend on where the other goose goes
def chance_to_collide(config, obs, action):
    next_head = translate_my_head(config, obs, action)
    goose_parts = {}
    for goose in obs.geese:
        gl = len(goose)
        for i, position in enumerate(goose):
            tt_leave = gl-i # ranges from 1 (tail) to goose length (head)
            # avoid following tail directly, in case the goose eats food (?)
            goose_parts[position] = tt_leave
    other_heads = [g[0] for i,g in enumerate(obs.geese) if (i != obs.index and len(g)>0)]
    
    # if I am walking right into somebody (including me), the chance is 1.
    if (next_head in goose_parts and goose_parts[next_head] > 0) or action.opposite()==prev_action:
        return 1
    
    headon_chances = []
    for h in other_heads:
        total_options = 0
        head_on = 0
        for a in Action:
            next_loc = translate(h, a, columns=config.columns, rows=config.rows)
            if (next_loc not in goose_parts or (goose_parts[next_loc] <= 0)): # goose may actually go there
                total_options += 1
                if next_loc == next_head:
                    head_on += 1
        if total_options > 0: # maybe that goose is in a dead end
            headon_chances.append(head_on/total_options)
    
    if len(headon_chances)==0:# maybe nobody has viable moves
        return 0
    
    return max(headon_chances)


# A very special q function to account for the fact that (for some reason) we gain an extra 100 points on the very first step
def is_first_step(config, obs, action):
    return 1 if step(obs)==1 else 0


# A function which returns a constant (like an intercept for linear regression)
def one(config, obs, action):
    return 1

q_weights={
    delta_length: 1,
    hunger_ttl: 80,
    uncontested_space: 20,
    nearest_food_dist: -1,
    chance_to_collide: -400,
    is_first_step: 100,
    one: 1,
}


# Given a state (defined by observation and configuration objects) and an action, 
# evaluate individual q-function components of q-value
def evaluate_q_function(obs, config, action):
    global q_weights
    qf_to_value = {}
    for q_func in q_weights:
        qf_to_value[q_func] = q_func(config, obs, action)
    return qf_to_value


# Given a state (defined by observation and configuration objects)
# Return what each q-function would evaluate to for *each* potential action we could take from this state.
def evaluate_q_functions(obs, config):
    action_to_qfs = {}
    for potential_action in Action:
        action_to_qfs[potential_action] = evaluate_q_function(obs, config, potential_action)
    return action_to_qfs


# Given data about a single q-function evaluation, as returned from evaluate_q_function,
# return the actual Q value of taking that action
def get_q_value(action_qf_values):
    q_value = 0
    for q_func in q_weights:
        q_value += q_weights[q_func]*action_qf_values[q_func]
    return q_value

# Given data about q-function evaluations (as returned by evaluate_q_functions) above,
# return the best action to take, and what its q-value would be
def get_best_q_value(action_to_qfs):
    global q_weights 
    options = []
    
    for potential_action in action_to_qfs:
        q_value = get_q_value(action_to_qfs[potential_action])
        options.append((potential_action, q_value))
    
    return max(options, key=lambda x: x[1])


# Commit to taking this particular action by updating global state (D:) as if it was already taken
def update_state(acion_taken):
    global prev_action
    prev_action = acion_taken


def agent(obs_dict, config_dict):
    obs= Observation(obs_dict)
    config = Configuration(config_dict)
    
    action_to_qfs = evaluate_q_functions(obs, config)
    best_action, best_Q_value = get_best_q_value(action_to_qfs) 
    
    update_state(best_action)
        
    return best_action.name