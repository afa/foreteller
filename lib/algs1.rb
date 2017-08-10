module Algs
 module Genetic
  module Classic
   def onemax(bitstring)
     sum = 0
     bitstring.size.times {|i| sum+=1 if bitstring[i].chr=='1'}
     return sum
   end
   def random_bitstring(num_bits)
     return (0...num_bits).inject(""){|s,i| s<<((rand<0.5) ? "1" : "0")}
   end
   def binary_tournament(pop)
     i, j = rand(pop.size), rand(pop.size)
     j = rand(pop.size) while j==i
     return (pop[i][:fitness] > pop[j][:fitness]) ? pop[i] : pop[j]
   end
   def point_mutation(bitstring, rate=1.0/bitstring.size)
     child = ""
      bitstring.size.times do |i|
        bit = bitstring[i].chr
        child << ((rand()<rate) ? ((bit=='1') ? "0" : "1") : bit)
     end
     return child
   end
   def crossover(parent1, parent2, rate)
     return ""+parent1 if rand()>=rate
     point = 1 + rand(parent1.size-2)
     return parent1[0...point]+parent2[point...(parent1.size)]
   end
   def reproduce(selected, pop_size, p_cross, p_mutation)

     children = []
     selected.each_with_index do |p1, i|
       p2 = (i.modulo(2)==0) ? selected[i+1] : selected[i-1]
       p2 = selected[0] if i == selected.size-1
       child = {}
       child[:bitstring] = crossover(p1[:bitstring], p2[:bitstring], p_cross)
       child[:bitstring] = point_mutation(child[:bitstring], p_mutation)
       children << child
       break if children.size >= pop_size
     end
     return children
   end
   def search(max_gens, num_bits, pop_size, p_crossover, p_mutation)
     population = Array.new(pop_size) do |i|
       {:bitstring=>random_bitstring(num_bits)}
     end
     population.each{|c| c[:fitness] = onemax(c[:bitstring])}
     best = population.sort{|x,y| y[:fitness] <=> x[:fitness]}.first
     max_gens.times do |gen|
       selected = Array.new(pop_size){|i| binary_tournament(population)}
       children = reproduce(selected, pop_size, p_crossover, p_mutation)
       children.each{|c| c[:fitness] = onemax(c[:bitstring])}
       children.sort!{|x,y| y[:fitness] <=> x[:fitness]}
       best = children.first if children.first[:fitness] >= best[:fitness]
       population = children
       puts " > gen #{gen}, best: #{best[:fitness]}, #{best[:bitstring]}"
       break if best[:fitness] == num_bits
     end
     return best
   end
   if __FILE__ == $0
     # problem configuration
     num_bits = 64
     # algorithm configuration
     max_gens = 100
     pop_size = 100
     p_crossover = 0.98
     p_mutation = 1.0/num_bits
     # execute the algorithm
     best = search(max_gens, num_bits, pop_size, p_crossover, p_mutation)
     puts "done! Solution: f=#{best[:fitness]}, s=#{best[:bitstring]}"
   end
  end
  module Programming
   def rand_in_bounds(min, max)
     return min + (max-min)*rand()
   end
   def print_program(node)
     return node if !node.kind_of?(Array)
     return "(#{node[0]} #{print_program(node[1])} #{print_program(node[2])})"
   end
   def eval_program(node, map)
     if !node.kind_of?(Array)
       return map[node].to_f if !map[node].nil?
       return node.to_f
     end
     arg1, arg2 = eval_program(node[1], map), eval_program(node[2], map)
     return 0 if node[0] === :/ and arg2 == 0.0
     return arg1.__send__(node[0], arg2)
   end
   def generate_random_program(max, funcs, terms, depth=0)
     if depth==max-1 or (depth>1 and rand()<0.1)
       t = terms[rand(terms.size)]
       return ((t=='R') ? rand_in_bounds(-5.0, +5.0) : t)
     end
     depth += 1
     arg1 = generate_random_program(max, funcs, terms, depth)
     arg2 = generate_random_program(max, funcs, terms, depth)
     return [funcs[rand(funcs.size)], arg1, arg2]
   end
   def count_nodes(node)
     return 1 if !node.kind_of?(Array)
     a1 = count_nodes(node[1])
     a2 = count_nodes(node[2])
     return a1+a2+1
   end
   def target_function(input)
     return input**2 + input + 1
   end
   def fitness(program, num_trials=20)
     sum_error = 0.0
     num_trials.times do |i|
       input = rand_in_bounds(-1.0, 1.0)
       error = eval_program(program, {'X'=>input}) - target_function(input)
       sum_error += error.abs
     end
     return sum_error / num_trials.to_f
   end
   def tournament_selection(pop, bouts)
     selected = Array.new(bouts){pop[rand(pop.size)]}
      selected.sort!{|x,y| x[:fitness]<=>y[:fitness]}
      return selected.first
    end
    def replace_node(node, replacement, node_num, cur_node=0)
      return [replacement,(cur_node+1)] if cur_node == node_num
      cur_node += 1
      return [node,cur_node] if !node.kind_of?(Array)
      a1, cur_node = replace_node(node[1], replacement, node_num, cur_node)
      a2, cur_node = replace_node(node[2], replacement, node_num, cur_node)
      return [[node[0], a1, a2], cur_node]
    end
    def copy_program(node)
      return node if !node.kind_of?(Array)
      return [node[0], copy_program(node[1]), copy_program(node[2])]
    end
    def get_node(node, node_num, current_node=0)
      return node,(current_node+1) if current_node == node_num
      current_node += 1
      return nil,current_node if !node.kind_of?(Array)
      a1, current_node = get_node(node[1], node_num, current_node)
      return a1,current_node if !a1.nil?
      a2, current_node = get_node(node[2], node_num, current_node)
      return a2,current_node if !a2.nil?
      return nil,current_node
    end
    def prune(node, max_depth, terms, depth=0)
      if depth == max_depth-1
        t = terms[rand(terms.size)]
        return ((t=='R') ? rand_in_bounds(-5.0, +5.0) : t)
      end
      depth += 1
      return node if !node.kind_of?(Array)
      a1 = prune(node[1], max_depth, terms, depth)
      a2 = prune(node[2], max_depth, terms, depth)
      return [node[0], a1, a2]
    end
    def crossover(parent1, parent2, max_depth, terms)
      pt1, pt2 = rand(count_nodes(parent1)-2)+1, rand(count_nodes(parent2)-2)+1
      tree1, c1 = get_node(parent1, pt1)
      tree2, c2 = get_node(parent2, pt2)
      child1, c1 = replace_node(parent1, copy_program(tree2), pt1)
      child1 = prune(child1, max_depth, terms)
      child2, c2 = replace_node(parent2, copy_program(tree1), pt2)
      child2 = prune(child2, max_depth, terms)
      return [child1, child2]
    end
    def mutation(parent, max_depth, functs, terms)
      random_tree = generate_random_program(max_depth/2, functs, terms)
      point = rand(count_nodes(parent))
      child, count = replace_node(parent, random_tree, point)
      child = prune(child, max_depth, terms)
      return child
    end
    def search(max_gens, pop_size, max_depth, bouts, p_repro, p_cross, p_mut,
         functs, terms)
      population = Array.new(pop_size) do |i|
        {:prog=>generate_random_program(max_depth, functs, terms)}
      end
      population.each{|c| c[:fitness] = fitness(c[:prog])}
      best = population.sort{|x,y| x[:fitness] <=> y[:fitness]}.first
      max_gens.times do |gen|
        children = []
        while children.size < pop_size
          operation = rand()
          p1 = tournament_selection(population, bouts)
          c1 = {}
          if operation < p_repro
            c1[:prog] = copy_program(p1[:prog])
          elsif operation < p_repro+p_cross
            p2 = tournament_selection(population, bouts)
            c2 = {}
            c1[:prog],c2[:prog] = crossover(p1[:prog], p2[:prog], max_depth,
                 terms)
            children << c2
          elsif operation < p_repro+p_cross+p_mut
            c1[:prog] = mutation(p1[:prog], max_depth, functs, terms)
          end
          children << c1 if children.size < pop_size
        end
        children.each{|c| c[:fitness] = fitness(c[:prog])}
        population = children
        population.sort!{|x,y| x[:fitness] <=> y[:fitness]}
        best = population.first if population.first[:fitness] <= best[:fitness]
        puts " > gen #{gen}, fitness=#{best[:fitness]}"
        break if best[:fitness] == 0
      end
      return best
    end
    if __FILE__ == $0
      # problem configuration
      terms = ['X', 'R']
      functs = [:+, :-, :*, :/]
      # algorithm configuration
      max_gens = 100
      max_depth = 7
      pop_size = 100
      bouts = 5
      p_repro = 0.08
      p_cross = 0.90
      p_mut = 0.02
      # execute the algorithm
      best = search(max_gens, pop_size, max_depth, bouts, p_repro, p_cross,
           p_mut, functs, terms)
      puts "done! Solution: f=#{best[:fitness]}, #{print_program(best[:prog])}"
    end
  end
  module EvolutionStrrategy
   def objective_function(vector)
     return vector.inject(0.0) {|sum, x| sum + (x ** 2.0)}
   end
   def random_vector(minmax)
     return Array.new(minmax.size) do |i|
       minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * rand())
     end
   end
   def random_gaussian(mean=0.0, stdev=1.0)
     u1 = u2 = w = 0
     begin
       u1 = 2 * rand() - 1
       u2 = 2 * rand() - 1
       w = u1 * u1 + u2 * u2
     end while w >= 1
     w = Math.sqrt((-2.0 * Math.log(w)) / w)
     return mean + (u2 * w) * stdev
   end
   def mutate_problem(vector, stdevs, search_space)
     child = Array(vector.size)
     vector.each_with_index do |v, i|
       child[i] = v + stdevs[i] * random_gaussian()
       child[i] = search_space[i][0] if child[i] < search_space[i][0]
       child[i] = search_space[i][1] if child[i] > search_space[i][1]
     end
     return child
   end
   def mutate_strategy(stdevs)
     tau = Math.sqrt(2.0*stdevs.size.to_f)**-1.0
     tau_p = Math.sqrt(2.0*Math.sqrt(stdevs.size.to_f))**-1.0
     child = Array.new(stdevs.size) do |i|
       stdevs[i] * Math.exp(tau_p*random_gaussian() + tau*random_gaussian())
     end
     return child
   end
   def mutate(par, minmax)
     child = {}
     child[:vector] = mutate_problem(par[:vector], par[:strategy], minmax)
     child[:strategy] = mutate_strategy(par[:strategy])
     return child
   end
   def init_population(minmax, pop_size)
     strategy = Array.new(minmax.size) do |i|
       [0, (minmax[i][1]-minmax[i][0]) * 0.05]
     end
     pop = Array.new(pop_size, {})
     pop.each_index do |i|
       pop[i][:vector] = random_vector(minmax)
       pop[i][:strategy] = random_vector(strategy)
     end
     pop.each{|c| c[:fitness] = objective_function(c[:vector])}
     return pop
   end
   def search(max_gens, search_space, pop_size, num_children)
     population = init_population(search_space, pop_size)
     best = population.sort{|x,y| x[:fitness] <=> y[:fitness]}.first
     max_gens.times do |gen|
       children = Array.new(num_children) do |i|
         mutate(population[i], search_space)
       end
       children.each{|c| c[:fitness] = objective_function(c[:vector])}
       union = children+population
       union.sort!{|x,y| x[:fitness] <=> y[:fitness]}
       best = union.first if union.first[:fitness] < best[:fitness]
       population = union.first(pop_size)
       puts " > gen #{gen}, fitness=#{best[:fitness]}"
     end
     return best
   end
   if __FILE__ == $0
     # problem configuration
     problem_size = 2
     search_space = Array.new(problem_size) {|i| [-5, +5]}
     # algorithm configuration
     max_gens = 100
     pop_size = 30
     num_children = 20
     # execute the algorithm
     best = search(max_gens, search_space, pop_size, num_children)
     puts "done! Solution: f=#{best[:fitness]}, s=#{best[:vector].inspect}"
   end
  end
  module DifferentialEvolution
   def objective_function(vector)
     return vector.inject(0.0) {|sum, x| sum + (x ** 2.0)}
   end
   def random_vector(minmax)
     return Array.new(minmax.size) do |i|
       minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * rand())
     end
   end
   def de_rand_1_bin(p0, p1, p2, p3, f, cr, search_space)
     sample = {:vector=>Array.new(p0[:vector].size)}
     cut = rand(sample[:vector].size-1) + 1
     sample[:vector].each_index do |i|
       sample[:vector][i] = p0[:vector][i]
       if (i==cut or rand() < cr)
         v = p3[:vector][i] + f * (p1[:vector][i] - p2[:vector][i])
         v = search_space[i][0] if v < search_space[i][0]
         v = search_space[i][1] if v > search_space[i][1]
         sample[:vector][i] = v
       end
     end
     return sample
   end
   def select_parents(pop, current)
     p1, p2, p3 = rand(pop.size), rand(pop.size), rand(pop.size)
     p1 = rand(pop.size) until p1 != current
     p2 = rand(pop.size) until p2 != current and p2 != p1
     p3 = rand(pop.size) until p3 != current and p3 != p1 and p3 != p2
     return [p1,p2,p3]
   end
   def create_children(pop, minmax, f, cr)
     children = []
     pop.each_with_index do |p0, i|
       p1, p2, p3 = select_parents(pop, i)
       children << de_rand_1_bin(p0, pop[p1], pop[p2], pop[p3], f, cr, minmax)
     end
     return children
   end
   def select_population(parents, children)
     return Array.new(parents.size) do |i|
       (children[i][:cost]<=parents[i][:cost]) ? children[i] : parents[i]
     end
   end
   def search(max_gens, search_space, pop_size, f, cr)
     pop = Array.new(pop_size) {|i| {:vector=>random_vector(search_space)}}
     pop.each{|c| c[:cost] = objective_function(c[:vector])}
     best = pop.sort{|x,y| x[:cost] <=> y[:cost]}.first
     max_gens.times do |gen|
       children = create_children(pop, search_space, f, cr)
       children.each{|c| c[:cost] = objective_function(c[:vector])}
       pop = select_population(pop, children)
       pop.sort!{|x,y| x[:cost] <=> y[:cost]}
       best = pop.first if pop.first[:cost] < best[:cost]
       puts " > gen #{gen+1}, fitness=#{best[:cost]}"
     end
     return best
   end
   if __FILE__ == $0
     # problem configuration
     problem_size = 3
     search_space = Array.new(problem_size) {|i| [-5, +5]}
     # algorithm configuration
     max_gens = 200
     pop_size = 10*problem_size
     weightf = 0.8
     crossf = 0.9
     # execute the algorithm
     best = search(max_gens, search_space, pop_size, weightf, crossf)
     puts "done! Solution: f=#{best[:cost]}, s=#{best[:vector].inspect}"
   end
  end
  module EvolutionProgramming
   def objective_function(vector)
     return vector.inject(0.0) {|sum, x| sum + (x ** 2.0)}
   end
   def random_vector(minmax)
     return Array.new(minmax.size) do |i|
       minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * rand())
     end
   end
   def random_gaussian(mean=0.0, stdev=1.0)
     u1 = u2 = w = 0
     begin
       u1 = 2 * rand() - 1
       u2 = 2 * rand() - 1
       w = u1 * u1 + u2 * u2
     end while w >= 1
     w = Math.sqrt((-2.0 * Math.log(w)) / w)
     return mean + (u2 * w) * stdev
   end
   def mutate(candidate, search_space)
     child = {:vector=>[], :strategy=>[]}
     candidate[:vector].each_with_index do |v_old, i|
       s_old = candidate[:strategy][i]
       v = v_old + s_old * random_gaussian()
       v = search_space[i][0] if v < search_space[i][0]
       v = search_space[i][1] if v > search_space[i][1]
       child[:vector] << v
       child[:strategy] << s_old + random_gaussian() * s_old.abs**0.5
     end
     return child
   end
   def tournament(candidate, population, bout_size)
     candidate[:wins] = 0
     bout_size.times do |i|
       other = population[rand(population.size)]
       candidate[:wins] += 1 if candidate[:fitness] < other[:fitness]
     end
   end
   def init_population(minmax, pop_size)
     strategy = Array.new(minmax.size) do |i|
       [0, (minmax[i][1]-minmax[i][0]) * 0.05]
     end
     pop = Array.new(pop_size, {})
     pop.each_index do |i|
       pop[i][:vector] = random_vector(minmax)
       pop[i][:strategy] = random_vector(strategy)
     end
     pop.each{|c| c[:fitness] = objective_function(c[:vector])}
     return pop
   end
   def search(max_gens, search_space, pop_size, bout_size)
     population = init_population(search_space, pop_size)
     population.each{|c| c[:fitness] = objective_function(c[:vector])}
     best = population.sort{|x,y| x[:fitness] <=> y[:fitness]}.first
     max_gens.times do |gen|
       children = Array.new(pop_size) {|i| mutate(population[i], search_space)}
       children.each{|c| c[:fitness] = objective_function(c[:vector])}
       children.sort!{|x,y| x[:fitness] <=> y[:fitness]}
       best = children.first if children.first[:fitness] < best[:fitness]
       union = children+population
       union.each{|c| tournament(c, union, bout_size)}
       union.sort!{|x,y| y[:wins] <=> x[:wins]}
       population = union.first(pop_size)
       puts " > gen #{gen}, fitness=#{best[:fitness]}"
     end
     return best
   end
   if __FILE__ == $0
     # problem configuration
     problem_size = 2
     search_space = Array.new(problem_size) {|i| [-5, +5]}
     # algorithm configuration
     max_gens = 200
     pop_size = 100
     bout_size = 5
     # execute the algorithm
     best = search(max_gens, search_space, pop_size, bout_size)
     puts "done! Solution: f=#{best[:fitness]}, s=#{best[:vector].inspect}"
   end
  end

  module GrammaticaEvolution
   def binary_tournament(pop)
     i, j = rand(pop.size), rand(pop.size)
     j = rand(pop.size) while j==i
     return (pop[i][:fitness] < pop[j][:fitness]) ? pop[i] : pop[j]
   end
   def point_mutation(bitstring, rate=1.0/bitstring.size.to_f)
     child = ""
     bitstring.size.times do |i|
       bit = bitstring[i].chr
       child << ((rand()<rate) ? ((bit=='1') ? "0" : "1") : bit)
     end
     return child
   end
   def one_point_crossover(parent1, parent2, codon_bits, p_cross=0.30)
     return ""+parent1[:bitstring] if rand()>=p_cross
     cut = rand([parent1.size, parent2.size].min/codon_bits)
     cut *= codon_bits
     p2size = parent2[:bitstring].size
     return parent1[:bitstring][0...cut]+parent2[:bitstring][cut...p2size]
   end
   def codon_duplication(bitstring, codon_bits, rate=1.0/codon_bits.to_f)
     return bitstring if rand() >= rate
     codons = bitstring.size/codon_bits
     return bitstring + bitstring[rand(codons)*codon_bits, codon_bits]
   end
   def codon_deletion(bitstring, codon_bits, rate=0.5/codon_bits.to_f)
     return bitstring if rand() >= rate
     codons = bitstring.size/codon_bits
     off = rand(codons)*codon_bits
     return bitstring[0...off] + bitstring[off+codon_bits...bitstring.size]
   end
   def reproduce(selected, pop_size, p_cross, codon_bits)
     children = []
     selected.each_with_index do |p1, i|
       p2 = (i.modulo(2)==0) ? selected[i+1] : selected[i-1]
       p2 = selected[0] if i == selected.size-1
       child = {}
       child[:bitstring] = one_point_crossover(p1, p2, codon_bits, p_cross)
       child[:bitstring] = codon_deletion(child[:bitstring], codon_bits)
       child[:bitstring] = codon_duplication(child[:bitstring], codon_bits)
       child[:bitstring] = point_mutation(child[:bitstring])
       children << child
       break if children.size == pop_size
     end
     return children
   end
   def random_bitstring(num_bits)
     return (0...num_bits).inject(""){|s,i| s<<((rand<0.5) ? "1" : "0")}
   end
   def decode_integers(bitstring, codon_bits)
     ints = []
     (bitstring.size/codon_bits).times do |off|
       codon = bitstring[off*codon_bits, codon_bits]
       sum = 0
       codon.size.times do |i|
         sum += ((codon[i].chr=='1') ? 1 : 0) * (2 ** i);
       end
       ints << sum
     end
     return ints
   end
   def map(grammar, integers, max_depth)
     done, offset, depth = false, 0, 0
     symbolic_string = grammar["S"]
     begin
       done = true
       grammar.keys.each do |key|
         symbolic_string = symbolic_string.gsub(key) do |k|
           done = false
           set = (k=="EXP" && depth>=max_depth-1) ? grammar["VAR"] : grammar[k]
           integer = integers[offset].modulo(set.size)
           offset = (offset==integers.size-1) ? 0 : offset+1
           set[integer]
         end
       end
       depth += 1
     end until done
     return symbolic_string
   end
   def target_function(x)
     return x**4.0 + x**3.0 + x**2.0 + x
   end
   def sample_from_bounds(bounds)
     return bounds[0] + ((bounds[1] - bounds[0]) * rand())
    end
    def cost(program, bounds, num_trials=30)
      return 9999999 if program.strip == "INPUT"
      sum_error = 0.0
      num_trials.times do
        x = sample_from_bounds(bounds)
        expression = program.gsub("INPUT", x.to_s)
        begin score = eval(expression) rescue score = 0.0/0.0 end
        return 9999999 if score.nan? or score.infinite?
        sum_error += (score - target_function(x)).abs
      end
      return sum_error / num_trials.to_f
    end
    def evaluate(candidate, codon_bits, grammar, max_depth, bounds)
      candidate[:integers] = decode_integers(candidate[:bitstring], codon_bits)
      candidate[:program] = map(grammar, candidate[:integers], max_depth)
      candidate[:fitness] = cost(candidate[:program], bounds)
    end
    def search(max_gens, pop_size, codon_bits, num_bits, p_cross, grammar,
         max_depth, bounds)
      pop = Array.new(pop_size) {|i| {:bitstring=>random_bitstring(num_bits)}}
      pop.each{|c| evaluate(c,codon_bits, grammar, max_depth, bounds)}
      best = pop.sort{|x,y| x[:fitness] <=> y[:fitness]}.first
      max_gens.times do |gen|
        selected = Array.new(pop_size){|i| binary_tournament(pop)}
        children = reproduce(selected, pop_size, p_cross,codon_bits)
        children.each{|c| evaluate(c, codon_bits, grammar, max_depth, bounds)}
        children.sort!{|x,y| x[:fitness] <=> y[:fitness]}
        best = children.first if children.first[:fitness] <= best[:fitness]
        pop=(children+pop).sort{|x,y| x[:fitness]<=>y[:fitness]}.first(pop_size)
        puts " > gen=#{gen}, f=#{best[:fitness]}, s=#{best[:bitstring]}"
        break if best[:fitness] == 0.0
      end
      return best
    end
    if __FILE__ == $0
      # problem configuration
      grammar = {"S"=>"EXP",
        "EXP"=>[" EXP BINARY EXP ", " (EXP BINARY EXP) ", " VAR "],
        "BINARY"=>["+", "-", "/", "*" ],
        "VAR"=>["INPUT", "1.0"]}
      bounds = [1, 10]
      # algorithm configuration
      max_depth = 7
      max_gens = 50
      pop_size = 100
      codon_bits = 4
      num_bits = 10*codon_bits
      p_cross = 0.30
      # execute the algorithm
      best = search(max_gens, pop_size, codon_bits, num_bits, p_cross, grammar,
           max_depth, bounds)
      puts "done! Solution: f=#{best[:fitness]}, s=#{best[:program]}"
    end
  end
  module GeneExpressionProgramming
   def binary_tournament(pop)
     i, j = rand(pop.size), rand(pop.size)
     return (pop[i][:fitness] < pop[j][:fitness]) ? pop[i] : pop[j]
   end
   def point_mutation(grammar, genome, head_length, rate=1.0/genome.size.to_f)
     child =""
     genome.size.times do |i|
       bit = genome[i].chr
       if rand() < rate
         if i < head_length
           selection = (rand() < 0.5) ? grammar["FUNC"]: grammar["TERM"]
           bit = selection[rand(selection.size)]
         else
           bit = grammar["TERM"][rand(grammar["TERM"].size)]
         end
       end
       child << bit
     end
     return child
   end
   def crossover(parent1, parent2, rate)
     return ""+parent1 if rand()>=rate
     child = ""
     parent1.size.times do |i|
       child << ((rand()<0.5) ? parent1[i] : parent2[i])
     end
     return child
   end
   def reproduce(grammar, selected, pop_size, p_crossover, head_length)
     children = []
     selected.each_with_index do |p1, i|
       p2 = (i.modulo(2)==0) ? selected[i+1] : selected[i-1]
       p2 = selected[0] if i == selected.size-1
       child = {}
       child[:genome] = crossover(p1[:genome], p2[:genome], p_crossover)
       child[:genome] = point_mutation(grammar, child[:genome], head_length)
       children << child
     end
     return children
   end
   def random_genome(grammar, head_length, tail_length)
     s = ""
     head_length.times do
       selection = (rand() < 0.5) ? grammar["FUNC"]: grammar["TERM"]
       s << selection[rand(selection.size)]
     end
     tail_length.times { s << grammar["TERM"][rand(grammar["TERM"].size)]}
     return s
   end
   def target_function(x)
     return x**4.0 + x**3.0 + x**2.0 + x
   end
   def sample_from_bounds(bounds)
     return bounds[0] + ((bounds[1] - bounds[0]) * rand())
   end
   def cost(program, bounds, num_trials=30)
     errors = 0.0
     num_trials.times do
       x = sample_from_bounds(bounds)
       expression, score = program.gsub("x", x.to_s), 0.0
       begin score = eval(expression) rescue score = 0.0/0.0 end
       return 9999999 if score.nan? or score.infinite?
       errors += (score - target_function(x)).abs
     end
     return errors / num_trials.to_f
   end
   def mapping(genome, grammar)
     off, queue = 0, []
     root = {}
     root[:node] = genome[off].chr; off+=1
     queue.push(root)
     while !queue.empty? do
       current = queue.shift
       if grammar["FUNC"].include?(current[:node])
         current[:left] = {}
         current[:left][:node] = genome[off].chr; off+=1
         queue.push(current[:left])
         current[:right] = {}
         current[:right][:node] = genome[off].chr; off+=1
         queue.push(current[:right])
       end
     end
     return root
   end
    def tree_to_string(exp)
      return exp[:node] if (exp[:left].nil? or exp[:right].nil?)
      left = tree_to_string(exp[:left])
      right = tree_to_string(exp[:right])
      return "(#{left} #{exp[:node]} #{right})"
    end
    def evaluate(candidate, grammar, bounds)
      candidate[:expression] = mapping(candidate[:genome], grammar)
      candidate[:program] = tree_to_string(candidate[:expression])
      candidate[:fitness] = cost(candidate[:program], bounds)
    end
    def search(grammar, bounds, h_length, t_length, max_gens, pop_size, p_cross)
      pop = Array.new(pop_size) do
        {:genome=>random_genome(grammar, h_length, t_length)}
      end
      pop.each{|c| evaluate(c, grammar, bounds)}
      best = pop.sort{|x,y| x[:fitness] <=> y[:fitness]}.first
      max_gens.times do |gen|
        selected = Array.new(pop){|i| binary_tournament(pop)}
        children = reproduce(grammar, selected, pop_size, p_cross, h_length)
        children.each{|c| evaluate(c, grammar, bounds)}
        children.sort!{|x,y| x[:fitness] <=> y[:fitness]}
        best = children.first if children.first[:fitness] <= best[:fitness]
        pop = (children+pop).first(pop_size)
        puts " > gen=#{gen}, f=#{best[:fitness]}, g=#{best[:genome]}"
      end
      return best
    end
    if __FILE__ == $0
      # problem configuration
      grammar = {"FUNC"=>["+","-","*","/"], "TERM"=>["x"]}
      bounds = [1.0, 10.0]
      # algorithm configuration
      h_length = 20
      t_length = h_length * (2-1) + 1
      max_gens = 150
      pop_size = 80
      p_cross = 0.85
      # execute the algorithm
      best = search(grammar, bounds, h_length, t_length, max_gens, pop_size,
           p_cross)
      puts "done! Solution: f=#{best[:fitness]}, program=#{best[:program]}"
    end
  end
  module LearningClassifierSystem
   def neg(bit)
     return (bit==1) ? 0 : 1
   end
   def target_function(s)
     ints = Array.new(6){|i| s[i].chr.to_i}
     x0,x1,x2,x3,x4,x5 = ints
     return neg(x0)*neg(x1)*x2 + neg(x0)*x1*x3 + x0*neg(x1)*x4 + x0*x1*x5
   end
   def new_classifier(condition, action, gen, p1=10.0, e1=0.0, f1=10.0)
     other = {}
     other[:condition],other[:action],other[:lasttime] = condition, action, gen
     other[:pred], other[:error], other[:fitness] = p1, e1, f1
     other[:exp], other[:setsize], other[:num] = 0.0, 1.0, 1.0
     return other
   end
   def copy_classifier(parent)
     copy = {}
     parent.keys.each do |k|
       copy[k] = (parent[k].kind_of? String) ? ""+parent[k] : parent[k]
     end
     copy[:num],copy[:exp] = 1.0, 0.0
     return copy
   end
   def random_bitstring(size=6)
     return (0...size).inject(""){|s,i| s+((rand<0.5) ? "1" : "0")}
   end
   def calculate_deletion_vote(classifier, pop, del_thresh, f_thresh=0.1)
     vote = classifier[:setsize] * classifier[:num]
     total = pop.inject(0.0){|s,c| s+c[:num]}
     avg_fitness = pop.inject(0.0){|s,c| s + (c[:fitness]/total)}
     derated = classifier[:fitness] / classifier[:num].to_f
     if classifier[:exp]>del_thresh and derated<(f_thresh*avg_fitness)
       return vote * (avg_fitness / derated)
     end
     return vote
   end
   def delete_from_pop(pop, pop_size, del_thresh=20.0)
     total = pop.inject(0) {|s,c| s+c[:num]}
     return if total <= pop_size
     pop.each {|c| c[:dvote] = calculate_deletion_vote(c, pop, del_thresh)}
     vote_sum = pop.inject(0.0) {|s,c| s+c[:dvote]}
     point = rand() * vote_sum
     vote_sum, index = 0.0, 0
     pop.each_with_index do |c,i|
       vote_sum += c[:dvote]
       if vote_sum >= point
         index = i
         break
       end
     end
     if pop[index][:num] > 1
       pop[index][:num] -= 1
     else
       pop.delete_at(index)
     end
   end
   def generate_random_classifier(input, actions, gen, rate=1.0/3.0)
     condition = ""
     input.size.times {|i| condition << ((rand<rate) ? '#' : input[i].chr)}
     action = actions[rand(actions.size)]
     return new_classifier(condition, action, gen)
   end
   def does_match?(input, condition)
     input.size.times do |i|
       return false if condition[i].chr!='#' and input[i].chr!=condition[i].chr
     end
     return true
   end
   def get_actions(pop)
     actions = []
      pop.each do |c|
        actions << c[:action] if !actions.include?(c[:action])
      end
      return actions
    end
    def generate_match_set(input, pop, all_actions, gen, pop_size)
      match_set = pop.select{|c| does_match?(input, c[:condition])}
      actions = get_actions(match_set)
      while actions.size < all_actions.size do
        remaining = all_actions - actions
        classifier = generate_random_classifier(input, remaining, gen)
        pop << classifier
        match_set << classifier
        delete_from_pop(pop, pop_size)
        actions << classifier[:action]
      end
      return match_set
    end
    def generate_prediction(match_set)
      pred = {}
      match_set.each do |classifier|
        key = classifier[:action]
        pred[key] = {:sum=>0.0,:count=>0.0,:weight=>0.0} if pred[key].nil?
        pred[key][:sum] += classifier[:pred]*classifier[:fitness]
        pred[key][:count] += classifier[:fitness]
      end
      pred.keys.each do |key|
        pred[key][:weight] = 0.0
        if pred[key][:count] > 0
          pred[key][:weight] = pred[key][:sum]/pred[key][:count]
        end
      end
      return pred
    end
    def select_action(predictions, p_explore=false)
      keys = Array.new(predictions.keys)
      return keys[rand(keys.size)] if p_explore
      keys.sort!{|x,y| predictions[y][:weight]<=>predictions[x][:weight]}
      return keys.first
    end
    def update_set(action_set, reward, beta=0.2)
      sum = action_set.inject(0.0) {|s,other| s+other[:num]}
      action_set.each do |c|
        c[:exp] += 1.0
        if c[:exp] < 1.0/beta
            c[:error] = (c[:error]*(c[:exp]-1.0)+(reward-c[:pred]).abs)/c[:exp]
            c[:pred] = (c[:pred] * (c[:exp]-1.0) + reward) / c[:exp]
            c[:setsize] = (c[:setsize]*(c[:exp]-1.0)+sum) / c[:exp]
        else
            c[:error] += beta * ((reward-c[:pred]).abs - c[:error])
            c[:pred] += beta * (reward-c[:pred])
            c[:setsize] += beta * (sum - c[:setsize])
        end
      end
    end
    def update_fitness(action_set, min_error=10, l_rate=0.2, alpha=0.1, v=-5.0)
      sum = 0.0
      acc = Array.new(action_set.size)
      action_set.each_with_index do |c,i|
        acc[i] = (c[:error]<min_error) ? 1.0 : alpha*(c[:error]/min_error)**v
        sum += acc[i] * c[:num].to_f
      end
      action_set.each_with_index do |c,i|
        c[:fitness] += l_rate * ((acc[i] * c[:num].to_f) / sum - c[:fitness])
      end
    end
    def can_run_genetic_algorithm(action_set, gen, ga_freq)
      return false if action_set.size <= 2
      total = action_set.inject(0.0) {|s,c| s+c[:lasttime]*c[:num]}
      sum = action_set.inject(0.0) {|s,c| s+c[:num]}
      return true if gen - (total/sum) > ga_freq
      return false
    end
    def binary_tournament(pop)
      i, j = rand(pop.size), rand(pop.size)
      j = rand(pop.size) while j==i
      return (pop[i][:fitness] > pop[j][:fitness]) ? pop[i] : pop[j]
    end
    def mutation(cl, action_set, input, rate=0.04)
      cl[:condition].size.times do |i|
        if rand() < rate
          cl[:condition][i] = (cl[:condition][i].chr=='#') ? input[i] : '#'
        end
      end
      if rand() < rate
        subset = action_set - [cl[:action]]
        cl[:action] = subset[rand(subset.size)]
      end
    end
    def uniform_crossover(parent1, parent2)
      child = ""
      parent1.size.times do |i|
        child << ((rand()<0.5) ? parent1[i].chr : parent2[i].chr)
      end
      return child
    end
    def insert_in_pop(cla, pop)
      pop.each do |c|
        if cla[:condition]==c[:condition] and cla[:action]==c[:action]
          c[:num] += 1
          return
        end
      end
      pop << cla
    end
    def crossover(c1, c2, p1, p2)
      c1[:condition] = uniform_crossover(p1[:condition], p2[:condition])
      c2[:condition] = uniform_crossover(p1[:condition], p2[:condition])
      c2[:pred] = c1[:pred] = (p1[:pred]+p2[:pred])/2.0
      c2[:error] = c1[:error] = 0.25*(p1[:error]+p2[:error])/2.0
      c2[:fitness] = c1[:fitness] = 0.1*(p1[:fitness]+p2[:fitness])/2.0
    end
    def run_ga(actions, pop, action_set, input, gen, pop_size, crate=0.8)
      p1, p2 = binary_tournament(action_set), binary_tournament(action_set)
      c1, c2 = copy_classifier(p1), copy_classifier(p2)
      crossover(c1, c2, p1, p2) if rand() < crate
      [c1,c2].each do |c|
        mutation(c, actions, input)
        insert_in_pop(c, pop)
      end
      while pop.inject(0) {|s,c| s+c[:num]} > pop_size
        delete_from_pop(pop, pop_size)
      end
    end
    def train_model(pop_size, max_gens, actions, ga_freq)
      pop, perf = [], []
      max_gens.times do |gen|
        explore = gen.modulo(2)==0
        input = random_bitstring()
        match_set = generate_match_set(input, pop, actions, gen, pop_size)
        pred_array = generate_prediction(match_set)
        action = select_action(pred_array, explore)
        reward = (target_function(input)==action.to_i) ? 1000.0 : 0.0
        if explore
          action_set = match_set.select{|c| c[:action]==action}
          update_set(action_set, reward)
          update_fitness(action_set)
          if can_run_genetic_algorithm(action_set, gen, ga_freq)
            action_set.each {|c| c[:lasttime] = gen}
            run_ga(actions, pop, action_set, input, gen, pop_size)
          end
        else
          e,a = (pred_array[action][:weight]-reward).abs, ((reward==1000.0)?1:0)
          perf << {:error=>e,:correct=>a}
          if perf.size >= 50
            err = (perf.inject(0){|s,x|s+x[:error]}/perf.size).round
            acc = perf.inject(0.0){|s,x|s+x[:correct]}/perf.size
            puts " >iter=#{gen+1} size=#{pop.size}, error=#{err}, acc=#{acc}"
            perf = []
          end
        end
      end
      return pop
    end
    def test_model(system, num_trials=50)
      correct = 0
      num_trials.times do
        input = random_bitstring()
        match_set = system.select{|c| does_match?(input, c[:condition])}
        pred_array = generate_prediction(match_set)
        action = select_action(pred_array, false)
        correct += 1 if target_function(input) == action.to_i
      end
      puts "Done! classified correctly=#{correct}/#{num_trials}"
      return correct
    end
    def execute(pop_size, max_gens, actions, ga_freq)
      system = train_model(pop_size, max_gens, actions, ga_freq)
      test_model(system)
      return system
    end
    if __FILE__ == $0
      # problem configuration
      all_actions = ['0', '1']
      # algorithm configuration
      max_gens, pop_size = 5000, 200
      ga_freq = 25
      # execute the algorithm
      execute(pop_size, max_gens, all_actions, ga_freq)
    end
  end
  module NondominatedSortingGeneticAlgorithm

  end
 end
end
