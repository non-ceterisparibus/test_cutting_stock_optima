from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, value


def make_naive_patterns(stocks, finish, MIN_MARGIN):
    """
    Generates patterns of feasible cuts from stock width to meet specified finish widths.

    Parameters:
    stocks (dict): A dictionary where keys are stock identifiers and values are dictionaries
                   with key 'length' representing the length of each stock.

    finish (dict): A dictionary where keys are finish identifiers and values are dictionaries
                   with key 'length' representing the required finish lengths.

    Returns:
    patterns (list): A list of dictionaries, where each dictionary represents a pattern of cuts.
                   Each pattern dictionary contains 'stock' (the stock identifier) and 'cuts'
                   (a dictionary where keys are finish identifiers and the value is the number
                   of cuts from the stock for each finish).
                   
                   Naive pattern with maximum number of cuts of each Finished Goods
                   that is closet to the required need_cut
                   and SUM(widths of FG) smaller Mother Coil width
    """

    patterns = []
    for f in finish:
        feasible = False
        for s in stocks:
            # max number of f that fit on s
            num_cuts_by_width = int((stocks[s]["width"]-MIN_MARGIN) / finish[f]["width"])
            # max number of f that satisfied the need cut WEIGHT BOUND
            num_cuts_by_weight = round((finish[f]["upper_bound"] * stocks[s]["width"] ) / (finish[f]["width"] * stocks[s]['weight']))
            # min of two max will satisfies both
            num_cuts = min(num_cuts_by_width, num_cuts_by_weight)

            # make pattern and add to list of patterns
            if num_cuts > 0:
                feasible = True
                cuts_dict = {key: 0 for key in finish.keys()}
                cuts_dict[f] = num_cuts
                trim_loss = stocks[s]['width'] - sum([finish[f]["width"] * cuts_dict[f] for f in finish.keys()])
                trim_loss_pct = round(trim_loss/stocks[s]['width'] * 100, 3)
                patterns.append({"stock": s, "cuts": cuts_dict, 'trim_loss':trim_loss, "trim_loss_pct": round(trim_loss_pct,2) })

        if not feasible:
            print(f"No feasible pattern was found for Stock {s} and FG {f}")

    return patterns

# NEW PATTERN - OBJECT FUNC KO WORK CHO WEIGHT
# GENERATE ON LINE CUT PATTERN - WEIGHT WONT BE CONSIDER, IN ADDITION, FIRST FIT STOCK WILL BE TAKEN
def new_pattern_problem(finish, width_s, ap_upper_bound, demand_duals, MIN_MARGIN):
    prob = LpProblem("NewPatternProblem", LpMaximize)

    # Decision variables - Pattern
    ap = {f: LpVariable(f"ap_{f}", 0, ap_upper_bound[f], cat="Integer") for f in finish.keys()}

    # Objective function
    # maximize marginal_cut:
    prob += lpSum(ap[f] * demand_duals[f] for f in finish.keys()), "MarginalCut"

    # Constraints - subject to stock_width - MIN MARGIN
    prob += lpSum(ap[f] * finish[f]["width"] for f in finish.keys()) <= width_s - MIN_MARGIN, "StockWidth_MinMargin"
    
    # Constraints - subject to trim loss 4% 
    prob += lpSum(ap[f] * finish[f]["width"] for f in finish.keys()) >= 0.96 * width_s , "StockWidth"

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs']))

    marg_cost = value(prob.objective)
    pattern = {f: int(ap[f].varValue) for f in finish.keys()}
    return marg_cost, pattern

# DUALITY
def generate_pattern_dual(stocks, finish, patterns, MIN_MARGIN):
    prob = LpProblem("GeneratePatternDual", LpMinimize)

    # Sets
    F = list(finish.keys())
    P = list(range(len(patterns)))

    # Parameters
    s = {p: patterns[p]["stock"] for p in range(len(patterns))}
    a = {(f, p): patterns[p]["cuts"][f] for p in P for f in F}
    demand_finish = {f: finish[f]["demand_slice"] for f in F}

    # Decision variables #relaxed integrality
    x = {p: LpVariable(f"x_{p}", 0, None, cat="Continuous") for p in P}

    # OBJECTIVE function minimize stock used:
    prob += lpSum(x[p] for p in P), "Cost"

    # Constraints
    for f in F:
        prob += lpSum(a[f, p] * x[p] for p in P) >= demand_finish[f], f"Demand_{f}"

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs']))

    # Extract dual values
    dual_values = {f: prob.constraints[f"Demand_{f}"].pi for f in F}

    ap_upper_bound = {f: max([patterns[i]['cuts'][f] for i,_ in enumerate(patterns)]) for f in finish.keys()}
    demand_duals = {f: dual_values[f] for f in F}

    marginal_values = {}
    pattern = {}
    for s in stocks.keys():
        marginal_values[s], pattern[s] = new_pattern_problem( #new pattern by line cut (trimloss), ignore weight
            finish, stocks[s]["width"], ap_upper_bound, demand_duals, MIN_MARGIN
        )

    s = max(marginal_values, key=marginal_values.get) # pick the first stock if having same width
    new_pattern = {"stock": s, "cuts": pattern[s]}
    return new_pattern

# CUT KNOWN WEIGHT PATTERNS
def cut_weight_patterns(stocks, finish, patterns):

    # Parameters - unit weight
    c = {p: stocks[pattern['stock']]["weight"]/stocks[pattern['stock']]["width"] for p, pattern in enumerate(patterns)}

    # Create a LP minimization problem
    prob = LpProblem("PatternCuttingProblem", LpMinimize)

    # Define variables
    x = {p: LpVariable(f"x_{p}", 0, 1, cat='Integer') for p in range(len(patterns))} # tu tach ta stock dung nhieu lan thanh 2 3 dong

    # Objective function: minimize total stock use
    prob += lpSum(x[p]* c[p] for p in range(len(patterns))), "TotalCost"

    # Constraints: meet demand for each finished part
    for f in finish:
        prob += lpSum(patterns[p]['cuts'][f] * finish[f]['width'] * x[p] * c[p] for p in range(len(patterns))) >= finish[f]['need_cut'], f"DemandWeight{f}"

    # Solve the problem
    prob.solve()

    # Extract results
    # Fix integer
    solution = [1 if (x[p].varValue > 0 and round(x[p].varValue)==0) else round(x[p].varValue) for p in range(len(patterns))]
    total_cost = sum(solution)

    return solution, total_cost

