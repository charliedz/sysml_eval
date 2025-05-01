# Created by cdz, derived from work done in CMSC723 Final Project, https://github.com/caustin5/nlp-final-project

import random
import json
import os
import argparse
from datasets import load_dataset

class Template:
    def __init__(self, marked_q, marked_a, dists, var_names):
        # self.template_id = template_id
        self.marked_q = marked_q
        self.marked_a = marked_a
        self.dists = dists
        self.var_names = var_names

    def generate_question(self):
        vals = {}
        for dist, var_name in zip(self.dists, self.var_names):
            vals[var_name] = dist(self, vals)
        question = self.marked_q.format(**vals)
        answer = self.marked_a.format(**vals)
        return question, answer

class Template_Generator:
    def __init__(self, names, names_male, names_female, male_family, female_family, gsm8k=None):
        self.names = names
        self.names_male = names_male
        self.names_female = names_female
        self.male_family = male_family
        self.female_family = female_family
        self.gsm8k = gsm8k
        self.templates = []

    def create_template(self, template_func):
        template_dict = template_func(self)
        marked_q = template_dict['marked_q']
        marked_a = template_dict['marked_a']
        dists = template_dict['dists']
        var_names = template_dict['var_names']
        self.templates.append(Template(marked_q, marked_a, dists, var_names))

    def generate_questions(self, template_index, num_questions):
        template = self.templates[template_index]
        questions = []
        for _ in range(num_questions):
            question, answer = template.generate_question()
            questions.append({'question': question, 'answer': answer})
        return questions

    def store_templates(self, directory, output_file, num_samples):
        os.chdir(directory)
        all_qa = []
        for i in range(len(self.templates)):
            print(f'working on prompt {i} of total {len(self.templates)}')
            qa_list = self.generate_questions(i, num_samples)
            all_qa.append(qa_list)
        with open(output_file, "w") as f:
            json.dump(all_qa, f, indent=4)
        print(f"Data successfully saved to {output_file}")

def template_1(generator):
    marked_q = '''{name:s} sold clips to {april_clips:d} of her friends in April, and then she sold {fraction:s} as many clips in May. How many clips did {name:s} sell altogether in April and May?'''
    marked_a = '''{name:s} sold {april_clips:d}/{divisor:d} = <<{april_clips:d}/{divisor:d}={may_clips:d}>>{may_clips:d} clips in May.
{name:s} sold {april_clips:d}+{may_clips:d} = <<{april_clips:d}+{may_clips:d}={total:d}>>{total:d} clips altogether in April and May.
#### {total:d}'''

    def dist_fraction(template, vals):
        return random.choice(["half", "a third", "a quarter"])

    def dist_divisor(template, vals):
        fraction_words = ["half", "a third", "a quarter"]
        return fraction_words.index(vals['fraction']) + 2

    def dist_april_clips(template, vals):
        return random.randint(1, 100) * vals['divisor']

    def dist_may_clips(template, vals):
        return vals['april_clips'] // vals['divisor']

    def dist_total(template, vals):
        return vals['april_clips'] + vals['may_clips']

    def dist_name(template, vals):
        return random.choice(generator.names_female)

    dists = [dist_fraction, dist_divisor, dist_april_clips, dist_may_clips, dist_total, dist_name]
    var_names = ['fraction', 'divisor', 'april_clips', 'may_clips', 'total', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_2(generator):
    marked_q = '''{name:s} earns ${hourly_rate:d} an hour for babysitting. Yesterday, she just did {minutes:d} minutes of babysitting. How much did she earn?'''
    marked_a = '''{name:s} earns {hourly_rate:d}/60 = $<<{hourly_rate:d}/60={per_minute:.1f}>>{per_minute:.1f} per minute.
Working {minutes:d} minutes, she earned {per_minute:.1f} x {minutes:d} = $<<{per_minute:.1f}*{minutes:d}={total:d}>>{total:d}.
#### {total:d}'''

    def dist_hourly_rate(template, vals):
        return random.randint(1, 6) * 6

    def dist_minutes(template, vals):
        return random.randint(1, 10) * 10

    def dist_per_minute(template, vals):
        return vals['hourly_rate'] / 60

    def dist_total(template, vals):
        return int(vals['per_minute'] * vals['minutes'])

    def dist_name(template, vals):
        return random.choice(generator.names_female)

    dists = [dist_hourly_rate, dist_minutes, dist_per_minute, dist_total, dist_name]
    var_names = ['hourly_rate', 'minutes', 'per_minute', 'total', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_3(generator):
    marked_q = '''{name:s} had {boxes:d} boxes of pencils with the same number of pencils in each box.
He kept {kept:d} pencils and shared the remaining pencils equally with his {friends:d} friends.
If his friends got {per_friend:d} pencils each, how many pencils are in each box?'''
    marked_a = '''{name:s} shared {friends:d} x {per_friend:d} = <<{friends:d}*{per_friend:d}={shared:d}>>{shared:d} pencils with his friends.
So, he had {kept:d} + {shared:d} = <<{kept:d}+{shared:d}={total:d}>>{total:d} pencils in all. Therefore,
each box had {total:d}/{boxes:d} = <<{total:d}/{boxes:d}={per_box:d}>>{per_box:d} pencils inside.
#### {per_box:d}'''

    def dist_kept(template, vals):
        return random.randint(5, 100)

    def dist_friends(template, vals):
        return random.randint(5, 100)

    def dist_per_friend(template, vals):
        return random.randint(5, 100)

    def dist_shared(template, vals):
        return vals['friends'] * vals['per_friend']

    def dist_total(template, vals):
        return vals['kept'] + vals['shared']

    def dist_boxes(template, vals):
        dividend = vals['total']
        divisors = [i for i in range(1, dividend + 1) if dividend % i == 0]
        return random.choice(divisors)

    def dist_per_box(template, vals):
        return vals['total'] // vals['boxes']

    def dist_name(template, vals):
        return random.choice(generator.names)

    dists = [dist_kept, dist_friends, dist_per_friend, dist_shared, dist_total, dist_boxes, dist_per_box, dist_name]
    var_names = ['kept', 'friends', 'per_friend', 'shared', 'total', 'boxes', 'per_box', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_4(generator):
    marked_q = '''{name:s} bought {ice_cream_cartons:d} cartons of ice cream and {yoghurt_cartons:d} cartons of frozen yoghurt.
Each carton of ice cream cost ${ice_cream_price:d} and each carton of frozen yoghurt cost ${yoghurt_price:d}.
How much more did {name:s} spend on ice cream than on frozen yoghurt?'''
    marked_a = '''The cost of the ice cream is {ice_cream_cartons:d} x ${ice_cream_price:d} = $<<{ice_cream_cartons:d}*{ice_cream_price:d}={ice_cream_cost:d}>>{ice_cream_cost:d}.
The cost of the frozen yoghurt is {yoghurt_cartons:d} x ${yoghurt_price:d} = $<<{yoghurt_cartons:d}*{yoghurt_price:d}={yoghurt_cost:d}>>{yoghurt_cost:d}.
{name:s} spent ${ice_cream_cost:d} - ${yoghurt_cost:d} = ${difference:d} more on ice cream than on frozen yogurt.
#### {difference:d}'''

    def dist_ice_cream_cartons(template, vals):
        return random.randint(5, 100)

    def dist_yoghurt_cartons(template, vals):
        return random.randint(5, vals['ice_cream_cartons'])

    def dist_ice_cream_price(template, vals):
        return random.randint(5, 100)

    def dist_yoghurt_price(template, vals):
        return random.randint(5, vals['ice_cream_price'])

    def dist_ice_cream_cost(template, vals):
        return vals['ice_cream_cartons'] * vals['ice_cream_price']

    def dist_yoghurt_cost(template, vals):
        return vals['yoghurt_cartons'] * vals['yoghurt_price']

    def dist_difference(template, vals):
        return vals['ice_cream_cost'] - vals['yoghurt_cost']

    def dist_name(template, vals):
        return random.choice(generator.names)

    dists = [dist_ice_cream_cartons, dist_yoghurt_cartons, dist_ice_cream_price, dist_yoghurt_price, 
             dist_ice_cream_cost, dist_yoghurt_cost, dist_difference, dist_name]
    var_names = ['ice_cream_cartons', 'yoghurt_cartons', 'ice_cream_price', 'yoghurt_price', 
                 'ice_cream_cost', 'yoghurt_cost', 'difference', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_5(generator):
    marked_q = '''{name:s} writes a {pages:d}-page letter to {friends:d} different friends {frequency:s} a {period:s}. How many pages does he write a year?'''
    marked_a = '''He writes each friend {pages:d}*{freq_num:d}=<<{pages:d}*{freq_num:d}={pages_per_friend:d}>>{pages_per_friend:d} pages a {period:s}.
So he writes {pages_per_friend:d}*{friends:d}=<<{pages_per_friend:d}*{friends:d}={total_pages_per_period:d}>>{total_pages_per_period:d} pages every {period:s}.
That means he writes {total_pages_per_period:d}*{periods_per_year:d}=<<{total_pages_per_period:d}*{periods_per_year:d}={total_pages_per_year:d}>>{total_pages_per_year:d} pages a year.
#### {total_pages_per_year:d}'''

    def dist_period(template, vals):
        return random.choice(["day", "week"])

    def dist_periods_per_year(template, vals):
        return 365 if vals['period'] == "day" else 52

    def dist_pages(template, vals):
        return random.randint(1, 6)

    def dist_friends(template, vals):
        return random.randint(1, 6)

    def dist_frequency(template, vals):
        return random.choice(["twice", "three times", "four times"])

    def dist_freq_num(template, vals):
        return {"twice": 2, "three times": 3, "four times": 4}[vals['frequency']]

    def dist_pages_per_friend(template, vals):
        return vals['pages'] * vals['freq_num']

    def dist_total_pages_per_period(template, vals):
        return vals['pages_per_friend'] * vals['friends']

    def dist_total_pages_per_year(template, vals):
        return vals['total_pages_per_period'] * vals['periods_per_year']

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    dists = [dist_period, dist_periods_per_year, dist_pages, dist_friends, dist_frequency,
             dist_freq_num, dist_pages_per_friend, dist_total_pages_per_period,
             dist_total_pages_per_year, dist_name]
    var_names = ['period', 'periods_per_year', 'pages', 'friends', 'frequency', 'freq_num',
                 'pages_per_friend', 'total_pages_per_period', 'total_pages_per_year', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_6(generator):
    marked_q = '''{name:s} has a garden with flowers. He planted plants of three different colors in it. {yellow_str:s} of them are yellow, and there are {purple_percent:d}% more of those in purple. There are only {green_percent:d}% as many green flowers as there are yellow and purple flowers. How many flowers does {name:s} have in his garden?'''
    marked_a = '''There are {purple_percent:d}/100 * {yellow:d} = <<{purple_percent:d}/100*{yellow:d}={purple_more:d}>>{purple_more:d} more purple flowers than yellow flowers.
So in {name:s}'s garden, there are {yellow:d} + {purple_more:d} = <<{yellow:d}+{purple_more:d}={purple:d}>>{purple:d} purple flowers.
Purple and yellow flowers sum up to {yellow:d} + {purple:d} = <<{yellow:d}+{purple:d}={yellow_purple_sum:d}>>{yellow_purple_sum:d} flowers.
That means in {name:s}'s garden there are {green_percent:d}/100 * {yellow_purple_sum:d} = <<{green_percent:d}/100*{yellow_purple_sum:d}={green:d}>>{green:d} green flowers.
So in total {name:s} has {yellow_purple_sum:d} + {green:d} = <<{yellow_purple_sum:d}+{green:d}={total:d}>>{total:d} plants in his garden.
#### {total:d}'''

    def dist_yellow_str(template, vals):
        return random.choice(["Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty"])

    def dist_yellow(template, vals):
        return (["Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty"].index(vals['yellow_str']) + 1) * 10

    def dist_purple_percent(template, vals):
        return (4 * random.randint(0, 2) + 2 * (vals['yellow'] // 10) % 2) * 10

    def dist_purple_more(template, vals):
        return (vals['purple_percent'] * vals['yellow']) // 100

    def dist_purple(template, vals):
        return vals['yellow'] + vals['purple_more']

    def dist_yellow_purple_sum(template, vals):
        return vals['yellow'] + vals['purple']

    def dist_green_percent(template, vals):
        return 25 * random.randint(1, 3)

    def dist_green(template, vals):
        return (vals['green_percent'] * vals['yellow_purple_sum']) // 100

    def dist_total(template, vals):
        return vals['yellow_purple_sum'] + vals['green']

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    dists = [dist_yellow_str, dist_yellow, dist_purple_percent, dist_purple_more, dist_purple,
             dist_yellow_purple_sum, dist_green_percent, dist_green, dist_total, dist_name]
    var_names = ['yellow_str', 'yellow', 'purple_percent', 'purple_more', 'purple',
                 'yellow_purple_sum', 'green_percent', 'green', 'total', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_7(generator):
    marked_q = '''{name:s} is wondering how much pizza he can eat in one day. He buys {large_pizzas:d} large pizzas and {small_pizzas:d} small pizzas. A large pizza has {large_slices:d} slices and a small pizza has {small_slices:d} slices. If he eats it all, how many pieces does he eat that day?'''
    marked_a = '''He eats {large_total:d} from the largest pizzas because {large_pizzas:d} x {large_slices:d} = <<{large_pizzas:d}*{large_slices:d}={large_total:d}>>{large_total:d}.
He eats {small_total:d} from the small pizza because {small_pizzas:d} x {small_slices:d} = <<{small_pizzas:d}*{small_slices:d}={small_total:d}>>{small_total:d}.
He eats {total_slices:d} pieces because {large_total:d} + {small_total:d} = <<{large_total:d}+{small_total:d}={total_slices:d}>>{total_slices:d}.
#### {total_slices:d}'''

    def dist_large_pizzas(template, vals):
        return random.randint(1, 5)

    def dist_small_pizzas(template, vals):
        return random.randint(1, 10)

    def dist_large_slices(template, vals):
        return random.randint(2, 5) * 4

    def dist_small_slices(template, vals):
        return random.randint(2, 4) * 2

    def dist_large_total(template, vals):
        return vals['large_pizzas'] * vals['large_slices']

    def dist_small_total(template, vals):
        return vals['small_pizzas'] * vals['small_slices']

    def dist_total_slices(template, vals):
        return vals['large_total'] + vals['small_total']

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    dists = [dist_large_pizzas, dist_small_pizzas, dist_large_slices, dist_small_slices,
             dist_large_total, dist_small_total, dist_total_slices, dist_name]
    var_names = ['large_pizzas', 'small_pizzas', 'large_slices', 'small_slices',
                 'large_total', 'small_total', 'total_slices', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_8(generator):
    marked_q = '''{name:s} created a care package to send to his {relation:s}, who was away at boarding school. {name:s} placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to {initial_weight:d} pounds. Then, he added enough brownies to cause the weight to {brownie_multiplier:s}. Next, he added another {additional_jelly_beans:d} pounds of jelly beans. And finally, he added enough gummy worms to {gummy_multiplier:s} the weight once again. What was the final weight of the box of goodies, in pounds?'''
    marked_a = '''To the initial {initial_weight:d} pounds of jelly beans, he added enough brownies to cause the weight to {brownie_multiplier:s}, bringing the weight to {initial_weight:d}*{brownie_factor:d}=<<{initial_weight:d}*{brownie_factor:d}={after_brownies:d}>>{after_brownies:d} pounds.
Next, he added another {additional_jelly_beans:d} pounds of jelly beans, bringing the weight to {after_brownies:d}+{additional_jelly_beans:d}=<<{after_brownies:d}+{additional_jelly_beans:d}={after_additional_jelly:d}>>{after_additional_jelly:d} pounds.
And finally, he added enough gummy worms to {gummy_multiplier:s} the weight once again, to a final weight of {after_additional_jelly:d}*{gummy_factor:d}=<<{after_additional_jelly:d}*{gummy_factor:d}={final_weight:d}>>{final_weight:d} pounds.
#### {final_weight:d}'''

    def dist_relation(template, vals):
        return random.choice(["brother", "sister", "cousin", "friend"])

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    def dist_initial_weight(template, vals):
        return random.randint(2, 5)

    def dist_brownie_multiplier(template, vals):
        return random.choice(["double", "triple", "quadruple"])

    def dist_brownie_factor(template, vals):
        return {"double": 2, "triple": 3, "quadruple": 4}[vals['brownie_multiplier']]

    def dist_gummy_multiplier(template, vals):
        return random.choice(["double", "triple", "quadruple"])

    def dist_gummy_factor(template, vals):
        return {"double": 2, "triple": 3, "quadruple": 4}[vals['gummy_multiplier']]

    def dist_after_brownies(template, vals):
        return vals['initial_weight'] * vals['brownie_factor']

    def dist_additional_jelly_beans(template, vals):
        return random.randint(2, 5)

    def dist_after_additional_jelly(template, vals):
        return vals['after_brownies'] + vals['additional_jelly_beans']

    def dist_final_weight(template, vals):
        return vals['after_additional_jelly'] * vals['gummy_factor']

    dists = [dist_relation, dist_name, dist_initial_weight, dist_brownie_multiplier,
             dist_brownie_factor, dist_gummy_multiplier, dist_gummy_factor,
             dist_after_brownies, dist_additional_jelly_beans, dist_after_additional_jelly,
             dist_final_weight]
    var_names = ['relation', 'name', 'initial_weight', 'brownie_multiplier', 'brownie_factor',
                 'gummy_multiplier', 'gummy_factor', 'after_brownies', 'additional_jelly_beans',
                 'after_additional_jelly', 'final_weight']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_9(generator):
    marked_q = '''{name:s} is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of ${budget:d} and spent ${shirt:d} on a button-up shirt, ${pants:d} on suit pants, ${coat:d} on a suit coat, ${socks:d} on socks, and ${belt:d} on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has ${remaining:d} left from her budget. How much did {name:s} pay for the shoes?'''
    marked_a = '''Let {var:s} be the amount {name:s} paid for the shoes.
She spent {var:s} + {shirt:d} + {pants:d} + {coat:d} + {socks:d} + {belt:d} = {var:s} + <<+{shirt:d}+{pants:d}+{coat:d}+{socks:d}+{belt:d}={total_spent_without_shoes:d}>>{total_spent_without_shoes:d}.
She used all but ${remaining:d} of her budget, so {var:s} + {total_spent_without_shoes:d} = {budget:d} - {remaining:d} = {amount_spent:d}.
Thus, {name:s} paid {var:s} = {amount_spent:d} - {total_spent_without_shoes:d} = $<<{amount_spent:d}-{total_spent_without_shoes:d}={shoes:d}>>{shoes:d} for the shoes.
#### {shoes:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names_female)

    def dist_var(template, vals):
        return random.choice(["A", "B", "C", "S", "X", "Y", "Z"])

    def dist_shirt(template, vals):
        return random.randint(20, 50)

    def dist_pants(template, vals):
        return random.randint(30, 80)

    def dist_coat(template, vals):
        return random.randint(30, 80)

    def dist_socks(template, vals):
        return random.randint(5, 20)

    def dist_belt(template, vals):
        return random.randint(10, 30)

    def dist_total_spent_without_shoes(template, vals):
        return vals['shirt'] + vals['pants'] + vals['coat'] + vals['socks'] + vals['belt']

    def dist_budget(template, vals):
        min_budget = (vals['total_spent_without_shoes'] // 10 + 4) * 10
        max_budget = (vals['total_spent_without_shoes'] // 10 + 10) * 10
        return random.randint(min_budget, max_budget)

    def dist_shoes(template, vals):
        return random.randint(1, vals['budget'] - vals['total_spent_without_shoes'] - 1)

    def dist_amount_spent(template, vals):
        return vals['total_spent_without_shoes'] + vals['shoes']

    def dist_remaining(template, vals):
        return vals['budget'] - vals['amount_spent']

    dists = [dist_name, dist_var, dist_shirt, dist_pants, dist_coat, dist_socks, dist_belt,
             dist_total_spent_without_shoes, dist_budget, dist_shoes, dist_amount_spent, dist_remaining]
    var_names = ['name', 'var', 'shirt', 'pants', 'coat', 'socks', 'belt',
                 'total_spent_without_shoes', 'budget', 'shoes', 'amount_spent', 'remaining']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_10(generator):
    marked_q = '''{name:s} makes ${hourly_rate:d}.00 an hour. If she works more than {base_hours:d} hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works {total_hours:d} hours every day for {days:d} days, how much money does she make?'''
    marked_a = '''She works {base_hours:d} hours a day for ${hourly_rate:d} per hour so she makes {base_hours:d}*{hourly_rate:d} = $<<{base_hours:d}*{hourly_rate:d}={base_pay:d}.00>>{base_pay:d}.00 per {base_hours:d}-hour shift.
She works {total_hours:d} hours a day and anything over {base_hours:d} hours is eligible for overtime, so she gets {total_hours:d}-{base_hours:d} = <<{total_hours:d}-{base_hours:d}={overtime_hours:d}>>{overtime_hours:d} hours of overtime.
Overtime is calculated as time and a half so and she makes ${hourly_rate:d}/hour so her overtime pay is {hourly_rate:d}*.5 = $<<{hourly_rate:d}*.5={overtime_add:d}.00>>{overtime_add:d}.00.
Her overtime pay is {hourly_rate:d}+{overtime_add:d} = $<<{hourly_rate:d}+{overtime_add:d}={overtime_rate:d}.00>>{overtime_rate:d}.00.
Her base pay is ${base_pay:d}.00 per {base_hours:d}-hour shift and she works {days:d} days and makes {days:d} * ${base_pay:d} = $<<{base_pay:d}*{days:d}={total_base_pay:d}>>{total_base_pay:d}.00.
Her overtime pay is ${overtime_rate:d}.00 per hour and she works {overtime_hours:d} hours of overtime per day and makes {overtime_rate:d}*{overtime_hours:d} = $<<{overtime_rate:d}*{overtime_hours:d}={daily_overtime_pay:d}.00>>{daily_overtime_pay:d}.00 in overtime pay.
{overtime_hours:d} hours of overtime pay for {days:d} days means she makes {daily_overtime_pay:d}*{days:d} = ${total_overtime_pay:d}.00.
In {days:d} days her base pay is ${total_base_pay:d}.00 and she makes ${total_overtime_pay:d}.00 in overtime pay so she makes ${total_base_pay:d} + ${total_overtime_pay:d} = $<<{total_base_pay:d}+{total_overtime_pay:d}={total_pay:d}.00>>{total_pay:d}.00.
#### {total_pay:d}'''

    def dist_hourly_rate(template, vals):
        return random.randint(6, 12) * 2

    def dist_base_hours(template, vals):
        return random.randint(6, 10)

    def dist_total_hours(template, vals):
        return random.randint(vals['base_hours'] + 1, vals['base_hours'] + 5)

    def dist_days(template, vals):
        return random.randint(2, 10)

    def dist_overtime_hours(template, vals):
        return vals['total_hours'] - vals['base_hours']

    def dist_base_pay(template, vals):
        return vals['hourly_rate'] * vals['base_hours']

    def dist_total_base_pay(template, vals):
        return vals['base_pay'] * vals['days']

    def dist_overtime_add(template, vals):
        return vals['hourly_rate'] // 2

    def dist_overtime_rate(template, vals):
        return vals['hourly_rate'] + vals['overtime_add']

    def dist_daily_overtime_pay(template, vals):
        return vals['overtime_rate'] * vals['overtime_hours']

    def dist_total_overtime_pay(template, vals):
        return vals['daily_overtime_pay'] * vals['days']

    def dist_total_pay(template, vals):
        return vals['total_base_pay'] + vals['total_overtime_pay']

    def dist_name(template, vals):
        return random.choice(generator.names_female)

    dists = [dist_hourly_rate, dist_base_hours, dist_total_hours, dist_days, dist_overtime_hours,
             dist_base_pay, dist_total_base_pay, dist_overtime_add, dist_overtime_rate,
             dist_daily_overtime_pay, dist_total_overtime_pay, dist_total_pay, dist_name]
    var_names = ['hourly_rate', 'base_hours', 'total_hours', 'days', 'overtime_hours', 'base_pay',
                 'total_base_pay', 'overtime_add', 'overtime_rate', 'daily_overtime_pay',
                 'total_overtime_pay', 'total_pay', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_11(generator):
    marked_q = '''A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed {total_people:d} people. Ships have been built larger over time, so each new ship has {multiplier:s} as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?'''
    marked_a = '''Let {var:s} be the number of people on the first hundred years’ ship.
The second hundred years’ ship had {multiplier:s} as many as the first, so it had {factor:d}{var:s} people.
The third hundred years’ ship had {multiplier:s} as many as the second, so it had {factor:d} * {factor:d}{var:s} = <<{factor:d}*{factor:d}={factor_squared:d}>>{factor_squared:d}{var:s} people.
All the ships had {var:s} + {factor:d}{var:s} + {factor_squared:d}{var:s} = {total_factor:d}{var:s} = {total_people:d} people.
Thus, the ship that the monster ate in the first hundred years had {var:s} = {total_people:d} / {total_factor:d} = <<{total_people:d}/{total_factor:d}={first_ship:d}>>{first_ship:d} people on it.
#### {first_ship:d}'''

    def dist_multiplier(template, vals):
        return random.choice(["twice", "three times", "four times"])

    def dist_factor(template, vals):
        return {"twice": 2, "three times": 3, "four times": 4}[vals['multiplier']]

    def dist_factor_squared(template, vals):
        return vals['factor'] * vals['factor']

    def dist_total_factor(template, vals):
        return 1 + vals['factor'] + vals['factor_squared']

    def dist_total_people(template, vals):
        return random.randint(10, 100) * vals['total_factor']

    def dist_first_ship(template, vals):
        return vals['total_people'] // vals['total_factor']

    def dist_var(template, vals):
        return random.choice(["A", "B", "C", "S", "X", "Y", "Z"])

    dists = [dist_multiplier, dist_factor, dist_factor_squared, dist_total_factor,
             dist_total_people, dist_first_ship, dist_var]
    var_names = ['multiplier', 'factor', 'factor_squared', 'total_factor', 'total_people',
                 'first_ship', 'var']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_12(generator):
    marked_q = '''{name:s} is buying a new pair of shoes that costs ${shoes_cost:d}. He has been saving up his money each month for the past {months_str:s} months. He gets a ${allowance:d} allowance a month. He also mows lawns and shovels driveways. He charges ${mow_rate:d} to mow a lawn and ${shovel_rate:d} to shovel. After buying the shoes, he has ${remaining:d} in change. If he mows {lawns:d} lawns, how many driveways did he shovel?'''
    marked_a = '''He saved up ${total_saved:d} total because {shoes_cost:d} + {remaining:d} = <<{shoes_cost:d}+{remaining:d}={total_saved:d}>>{total_saved:d}.
He saved ${allowance_total:d} from his allowance because {months:d} x {allowance:d} = <<{months:d}*{allowance:d}={allowance_total:d}>>{allowance_total:d}.
He earned ${mow_earnings:d} mowing lawns because {lawns:d} x {mow_rate:d} = <<{lawns:d}*{mow_rate:d}={mow_earnings:d}>>{mow_earnings:d}.
He earned ${shovel_earnings:d} shoveling driveways because {total_saved:d} - {mow_earnings:d} - {allowance_total:d} = <<{total_saved:d}-{mow_earnings:d}-{allowance_total:d}={shovel_earnings:d}>>{shovel_earnings:d}.
He shoveled {driveways:d} driveways because {shovel_earnings:d} / {shovel_rate:d} = <<{shovel_earnings:d}/{shovel_rate:d}={driveways:d}>>{driveways:d}.
#### {driveways:d}'''

    def dist_months_str(template, vals):
        return random.choice(["two", "three", "four", "five", "six"])

    def dist_months(template, vals):
        return {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}[vals['months_str']]

    def dist_allowance(template, vals):
        return random.randint(5, 15)

    def dist_mow_rate(template, vals):
        return random.randint(10, 25)

    def dist_lawns(template, vals):
        return random.randint(2, 6)

    def dist_mow_earnings(template, vals):
        return vals['lawns'] * vals['mow_rate']

    def dist_allowance_total(template, vals):
        return vals['months'] * vals['allowance']

    def dist_shovel_rate(template, vals):
        return random.randint(5, 25)

    def dist_driveways(template, vals):
        return random.randint(2, 6)

    def dist_shovel_earnings(template, vals):
        return vals['shovel_rate'] * vals['driveways']

    def dist_total_saved(template, vals):
        return vals['shovel_earnings'] + vals['mow_earnings'] + vals['allowance_total']

    def dist_remaining(template, vals):
        return random.randint(1, 20)

    def dist_shoes_cost(template, vals):
        return vals['total_saved'] - vals['remaining']

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    dists = [dist_months_str, dist_months, dist_allowance, dist_mow_rate, dist_lawns,
             dist_mow_earnings, dist_allowance_total, dist_shovel_rate, dist_driveways,
             dist_shovel_earnings, dist_total_saved, dist_remaining, dist_shoes_cost, dist_name]
    var_names = ['months_str', 'months', 'allowance', 'mow_rate', 'lawns', 'mow_earnings',
                 'allowance_total', 'shovel_rate', 'driveways', 'shovel_earnings',
                 'total_saved', 'remaining', 'shoes_cost', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_13(generator):
    marked_q = '''{name:s} has {mango_trees:d} mango trees on his farm. He also has {less_than:d} less than {fraction:s} as many coconut trees as mango trees. How many trees does {name:s} have in all on his farm?'''
    marked_a = '''{fraction:s} of the number of {name:s}'s mango trees is {mango_trees:d}/{denominator:d} = <<{mango_trees:d}/{denominator:d}={fraction_mango:d}>>{fraction_mango:d} trees.
So {name:s} has {fraction_mango:d} - {less_than:d} = <<{fraction_mango:d}-{less_than:d}={coconut_trees:d}>>{coconut_trees:d} coconut trees.
Therefore, {name:s} has {mango_trees:d} + {coconut_trees:d} = <<{mango_trees:d}+{coconut_trees:d}={total_trees:d}>>{total_trees:d} trees on his farm.
#### {total_trees:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    def dist_fraction(template, vals):
        return random.choice(["half", "a third", "a quarter"])

    def dist_denominator(template, vals):
        return {"half": 2, "a third": 3, "a quarter": 4}[vals['fraction']]

    def dist_mango_trees(template, vals):
        return random.randint(2, 30) * vals['denominator']

    def dist_fraction_mango(template, vals):
        return vals['mango_trees'] // vals['denominator']

    def dist_less_than(template, vals):
        return random.randint(1, vals['fraction_mango'] - 1)

    def dist_coconut_trees(template, vals):
        return vals['fraction_mango'] - vals['less_than']

    def dist_total_trees(template, vals):
        return vals['mango_trees'] + vals['coconut_trees']

    dists = [dist_name, dist_fraction, dist_denominator, dist_mango_trees, dist_fraction_mango,
             dist_less_than, dist_coconut_trees, dist_total_trees]
    var_names = ['name', 'fraction', 'denominator', 'mango_trees', 'fraction_mango',
                 'less_than', 'coconut_trees', 'total_trees']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_14(generator):
    marked_q = '''{name:s} will serve charcuterie at his dinner party. He buys {cheddar_pounds:d} pounds of cheddar cheese for ${cheddar_cost:d}, a pound of cream cheese that cost {fraction:s} the price of the cheddar cheese, and a pack of cold cuts that cost {multiplier:s} the price of the cheddar cheese. How much does he spend on the ingredients?'''
    marked_a = '''A pound of cream cheese cost ${cheddar_cost:d} / {denominator:d} = $<<{cheddar_cost:d}/{denominator:d}={cream_cheese_cost:d}>>{cream_cheese_cost:d}.
A pack of cold cuts cost ${cheddar_cost:d} x {factor:d} = $<<{cheddar_cost:d}*{factor:d}={cold_cuts_cost:d}>>{cold_cuts_cost:d}.
{name:s} spent ${cheddar_cost:d} + ${cream_cheese_cost:d} + ${cold_cuts_cost:d} = $<<{cheddar_cost:d}+{cream_cheese_cost:d}+{cold_cuts_cost:d}={total_cost:d}>>{total_cost:d} on the ingredients.
#### {total_cost:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    def dist_fraction(template, vals):
        return random.choice(["half", "a third", "a quarter"])

    def dist_denominator(template, vals):
        return {"half": 2, "a third": 3, "a quarter": 4}[vals['fraction']]

    def dist_cream_cheese_cost(template, vals):
        return random.randint(1, 10)

    def dist_cheddar_cost(template, vals):
        return vals['denominator'] * vals['cream_cheese_cost']

    def dist_cheddar_pounds(template, vals):
        return random.randint(2, 8)

    def dist_multiplier(template, vals):
        return random.choice(["twice", "three times", "four times"])

    def dist_factor(template, vals):
        return {"twice": 2, "three times": 3, "four times": 4}[vals['multiplier']]

    def dist_cold_cuts_cost(template, vals):
        return vals['cheddar_cost'] * vals['factor']

    def dist_total_cost(template, vals):
        return vals['cheddar_cost'] * vals['cheddar_pounds'] + vals['cream_cheese_cost'] + vals['cold_cuts_cost']

    dists = [dist_name, dist_fraction, dist_denominator, dist_cream_cheese_cost, dist_cheddar_cost,
             dist_cheddar_pounds, dist_multiplier, dist_factor, dist_cold_cuts_cost, dist_total_cost]
    var_names = ['name', 'fraction', 'denominator', 'cream_cheese_cost', 'cheddar_cost',
                 'cheddar_pounds', 'multiplier', 'factor', 'cold_cuts_cost', 'total_cost']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_15(generator):
    marked_q = '''{name:s} can read {pages_per_period:d} pages of a book in {minutes_per_period:d} minutes. How many hours will it take her to read {total_pages:d} pages?'''
    marked_a = '''In one hour, there are {periods_per_hour:d} sets of {minutes_per_period:d} minutes.
So, {name:s} can read {pages_per_period:d} x {periods_per_hour:d} = <<{pages_per_period:d}*{periods_per_hour:d}={pages_per_hour:d}>>{pages_per_hour:d} pages in an hour.
It will take her {total_pages:d}/{pages_per_hour:d} = <<{total_pages:d}/{pages_per_hour:d}={hours:d}>>{hours:d} hours to read {total_pages:d} pages.
#### {hours:d}'''

    def dist_minutes_per_period(template, vals):
        return random.choice([5, 6, 10, 12, 15, 20, 30])

    def dist_periods_per_hour(template, vals):
        return 60 // vals['minutes_per_period']

    def dist_pages_per_period(template, vals):
        return random.randint(2, 10)

    def dist_pages_per_hour(template, vals):
        return vals['pages_per_period'] * vals['periods_per_hour']

    def dist_hours(template, vals):
        return random.randint(2, 8)

    def dist_total_pages(template, vals):
        return vals['pages_per_hour'] * vals['hours']

    def dist_name(template, vals):
        return random.choice(generator.names_female)

    dists = [dist_minutes_per_period, dist_periods_per_hour, dist_pages_per_period,
             dist_pages_per_hour, dist_hours, dist_total_pages, dist_name]
    var_names = ['minutes_per_period', 'periods_per_hour', 'pages_per_period', 'pages_per_hour',
                 'hours', 'total_pages', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_16(generator):
    marked_q = '''{name:s} creates a media empire. He creates a movie for ${movie_cost:d}. Each DVD cost ${dvd_cost:d} to make. He sells it for {sell_factor:.1f} times that much. He sells {dvds_per_day:d} movies a day for {days_per_week:d} days a week. How much profit does he make in {weeks:d} weeks?'''
    marked_a = '''He sold each DVD for {dvd_cost:d}*{sell_factor:.1f}=$<<{dvd_cost:d}*{sell_factor:.1f}={sell_price:d}>>{sell_price:d}.
So he makes a profit of {sell_price:d}-{dvd_cost:d}=$<<{sell_price:d}-{dvd_cost:d}={profit_per_dvd:d}>>{profit_per_dvd:d}.
So each day he makes a profit of {profit_per_dvd:d}*{dvds_per_day:d}=$<<{profit_per_dvd:d}*{dvds_per_day:d}={daily_profit:d}>>{daily_profit:d}.
So he makes {daily_profit:d}*{days_per_week:d}=$<<{daily_profit:d}*{days_per_week:d}={weekly_profit:d}>>{weekly_profit:d}.
He makes {weekly_profit:d}*{weeks:d}=$<<{weekly_profit:d}*{weeks:d}={total_profit_before_cost:d}>>{total_profit_before_cost:d}.
Then after the cost of creating the movie he has a profit of {total_profit_before_cost:d}-{movie_cost:d}=$<<{total_profit_before_cost:d}-{movie_cost:d}={final_profit:d}>>{final_profit:d}.
#### {final_profit:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    def dist_movie_cost(template, vals):
        return random.randint(1, 10) * 1000

    def dist_dvd_cost(template, vals):
        return random.randint(1, 5) * 2

    def dist_sell_factor(template, vals):
        return random.randint(1, 4) + 0.5

    def dist_dvds_per_day(template, vals):
        return random.randint(1, 10) * 100

    def dist_days_per_week(template, vals):
        return random.randint(2, 7)

    def dist_weeks(template, vals):
        return random.randint(5, 35)

    def dist_sell_price(template, vals):
        return int(vals['dvd_cost'] * vals['sell_factor'])

    def dist_profit_per_dvd(template, vals):
        return vals['sell_price'] - vals['dvd_cost']

    def dist_daily_profit(template, vals):
        return vals['profit_per_dvd'] * vals['dvds_per_day']

    def dist_weekly_profit(template, vals):
        return vals['daily_profit'] * vals['days_per_week']

    def dist_total_profit_before_cost(template, vals):
        return vals['weekly_profit'] * vals['weeks']

    def dist_final_profit(template, vals):
        return vals['total_profit_before_cost'] - vals['movie_cost']

    dists = [dist_name, dist_movie_cost, dist_dvd_cost, dist_sell_factor, dist_dvds_per_day,
             dist_days_per_week, dist_weeks, dist_sell_price, dist_profit_per_dvd,
             dist_daily_profit, dist_weekly_profit, dist_total_profit_before_cost, dist_final_profit]
    var_names = ['name', 'movie_cost', 'dvd_cost', 'sell_factor', 'dvds_per_day',
                 'days_per_week', 'weeks', 'sell_price', 'profit_per_dvd', 'daily_profit',
                 'weekly_profit', 'total_profit_before_cost', 'final_profit']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_17(generator):
    marked_q = '''The profit from a business transaction is shared among 2 business partners, {partner1:s} and {partner2:s} in the ratio {ratio1:d}:{ratio2:d} respectively. If {partner2:s} got ${partner2_share:d}, how much will {partner1:s} have after spending some of his share on a shirt that costs ${shirt_cost:d}?'''
    marked_a = '''According to the ratio, for every {ratio2:d} parts that {partner2:s} gets, {partner1:s} gets {ratio1:d} parts.
Since {partner2:s} got ${partner2_share:d}, each part is therefore ${partner2_share:d}/{ratio2:d} = $<<{partner2_share:d}/{ratio2:d}={part_value:d}>>{part_value:d}.
{partner1:s} will get {ratio1:d}*${part_value:d} = $<<{ratio1:d}*{part_value:d}={partner1_share:d}>>{partner1_share:d}.
After buying the shirt he will have ${partner1_share:d}-${shirt_cost:d} = $<<{partner1_share:d}-{shirt_cost:d}={remaining:d}>>{remaining:d} left.
#### {remaining:d}'''

    def dist_partner1(template, vals):
        return random.choice(generator.names_male)

    def dist_partner2(template, vals):
        available_names = [name for name in generator.names_male if name != vals['partner1']]
        return random.choice(available_names)

    def dist_ratio1(template, vals):
        return random.randint(2, 6)

    def dist_ratio2(template, vals):
        return random.randint(2, 6)

    def dist_part_value(template, vals):
        return random.randint(100, 1000)

    def dist_partner1_share(template, vals):
        return vals['ratio1'] * vals['part_value']

    def dist_partner2_share(template, vals):
        return vals['ratio2'] * vals['part_value']

    def dist_shirt_cost(template, vals):
        return random.randint(1, vals['partner1_share'] // 5)

    def dist_remaining(template, vals):
        return vals['partner1_share'] - vals['shirt_cost']

    dists = [dist_partner1, dist_partner2, dist_ratio1, dist_ratio2, dist_part_value,
             dist_partner1_share, dist_partner2_share, dist_shirt_cost, dist_remaining]
    var_names = ['partner1', 'partner2', 'ratio1', 'ratio2', 'part_value', 'partner1_share',
                 'partner2_share', 'shirt_cost', 'remaining']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_18(generator):
    marked_q = '''In a truck, there are {pink_hats:d} pink hard hats, {green_hats:d} green hard hats, and {yellow_hats:d} yellow hard hats. If {worker1:s} takes away {pink_taken1:d} pink hard hats, and {worker2:s} takes away {pink_taken2:d} pink hard hats and {multiplier:s} as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.'''
    marked_a = '''If there were {pink_hats:d} pink hard hats and {worker1:s} took away {pink_taken1:d} pink hard hats, the number of pink hard hats that remained is {pink_hats:d}-{pink_taken1:d} = <<{pink_hats:d}-{pink_taken1:d}={pink_after_worker1:d}>>{pink_after_worker1:d}.
{worker2:s} also took away {pink_taken2:d} pink hard hats, leaving {pink_after_worker1:d}-{pink_taken2:d} = <<{pink_after_worker1:d}-{pink_taken2:d}={pink_remaining:d}>>{pink_remaining:d} pink hard hats in the truck.
If {worker2:s} also took {multiplier:s} as many green hard hats as pink hard hats, he took {factor:d}*{pink_taken2:d} = <<{factor:d}*{pink_taken2:d}={green_taken:d}>>{green_taken:d} green hard hats.
The total number of green hard hats that remained in the truck is {green_hats:d}-{green_taken:d} = <<{green_hats:d}-{green_taken:d}={green_remaining:d}>>{green_remaining:d}.
In the truck, after some are taken, there were {green_remaining:d} green hard hats + {pink_remaining:d} pink hard hats = <<{green_remaining:d}+{pink_remaining:d}={green_pink_remaining:d}>>{green_pink_remaining:d} hard hats in the truck.
Altogether, {green_pink_remaining:d} green and pink hard hats + {yellow_hats:d} yellow hard hats = <<{green_pink_remaining:d}+{yellow_hats:d}={total_remaining:d}>>{total_remaining:d} hard hats remained in the truck.
#### {total_remaining:d}'''

    def dist_worker1(template, vals):
        return random.choice(generator.names_male)

    def dist_worker2(template, vals):
        available_names = [name for name in generator.names_male if name != vals['worker1']]
        return random.choice(available_names)

    def dist_multiplier(template, vals):
        return random.choice(["twice", "three times", "four times"])

    def dist_factor(template, vals):
        return {"twice": 2, "three times": 3, "four times": 4}[vals['multiplier']]

    def dist_pink_taken2(template, vals):
        return random.randint(2, 10)

    def dist_green_taken(template, vals):
        return vals['factor'] * vals['pink_taken2']

    def dist_green_hats(template, vals):
        # Ensure green_hats > green_taken to avoid negative green_remaining
        return random.randint(vals['green_taken'] + 1, 30)

    def dist_pink_hats(template, vals):
        # Ensure pink_hats >= pink_taken1 + pink_taken2, and max is 30
        min_pink_hats = vals['pink_taken2'] + 1  # Minimum to allow pink_taken1 >= 1
        return random.randint(min_pink_hats, 30)

    def dist_pink_taken1(template, vals):
        # pink_taken1 must leave enough for pink_taken2 (i.e., pink_after_worker1 >= pink_taken2)
        max_pink_taken1 = vals['pink_hats'] - vals['pink_taken2']
        return random.randint(1, max_pink_taken1)

    def dist_pink_after_worker1(template, vals):
        return vals['pink_hats'] - vals['pink_taken1']

    def dist_pink_remaining(template, vals):
        return vals['pink_after_worker1'] - vals['pink_taken2']

    def dist_yellow_hats(template, vals):
        return random.randint(2, 30)

    def dist_green_remaining(template, vals):
        return vals['green_hats'] - vals['green_taken']

    def dist_green_pink_remaining(template, vals):
        return vals['green_remaining'] + vals['pink_remaining']

    def dist_total_remaining(template, vals):
        return vals['green_pink_remaining'] + vals['yellow_hats']

    dists = [
        dist_worker1, dist_worker2, dist_multiplier, dist_factor, dist_pink_taken2,
        dist_green_taken, dist_green_hats, dist_pink_hats, dist_pink_taken1,
        dist_pink_after_worker1, dist_pink_remaining, dist_yellow_hats,
        dist_green_remaining, dist_green_pink_remaining, dist_total_remaining
    ]
    var_names = [
        'worker1', 'worker2', 'multiplier', 'factor', 'pink_taken2', 'green_taken',
        'green_hats', 'pink_hats', 'pink_taken1', 'pink_after_worker1', 'pink_remaining',
        'yellow_hats', 'green_remaining', 'green_pink_remaining', 'total_remaining'
    ]
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_19(generator):
    marked_q = '''It takes {name:s} {walk_time_str:s} to walk to work and {bike_time_str:s} to ride his bike to work. {name:s} walks to and from work {walk_days_str:s} a week and rides his bike to and from work {bike_days_str:s} a week. How many hours in total does he take to get to and from work a week with walking and biking?'''
    marked_a = '''{name:s} takes {walk_time:d}*{walk_days:d} = <<{walk_time:d}*{walk_days:d}={walk_to_work:d}>>{walk_to_work:d} hours a week to walk to work.
{name:s} takes {walk_to_work:d}*2 = <<{walk_to_work:d}*2={walk_total:d}>>{walk_total:d} hours a week to walk to and from work.
{name:s} takes {bike_time:d}*{bike_days:d} = <<{bike_time:d}*{bike_days:d}={bike_to_work:d}>>{bike_to_work:d} hours a week to bike to work.
{name:s} takes {bike_to_work:d}*2 = <<{bike_to_work:d}*2={bike_total:d}>>{bike_total:d} hours a week to bike to and from work.
In total, {name:s} takes {walk_total:d}+{bike_total:d} = <<{walk_total:d}+{bike_total:d}={total_time:d}>>{total_time:d} hours a week to go to and from work.
#### {total_time:d}'''

    def dist_walk_time_str(template, vals):
        return random.choice(["one hour", "two hours", "three hours", "four hours"])

    def dist_walk_time(template, vals):
        return {"one hour": 1, "two hours": 2, "three hours": 3, "four hours": 4}[vals['walk_time_str']]

    def dist_walk_days_str(template, vals):
        return random.choice(["once", "twice", "three times", "four times"])

    def dist_walk_days(template, vals):
        return {"once": 1, "twice": 2, "three times": 3, "four times": 4}[vals['walk_days_str']]

    def dist_walk_to_work(template, vals):
        return vals['walk_time'] * vals['walk_days']

    def dist_walk_total(template, vals):
        return vals['walk_to_work'] * 2

    def dist_bike_time_str(template, vals):
        return random.choice(["one hour", "two hours", "three hours", "four hours"])

    def dist_bike_time(template, vals):
        return {"one hour": 1, "two hours": 2, "three hours": 3, "four hours": 4}[vals['bike_time_str']]

    def dist_bike_days(template, vals):
        return 5 - vals['walk_days']

    def dist_bike_days_str(template, vals):
        return ["once", "twice", "three times", "four times"][vals['bike_days'] - 1]

    def dist_bike_to_work(template, vals):
        return vals['bike_time'] * vals['bike_days']

    def dist_bike_total(template, vals):
        return vals['bike_to_work'] * 2

    def dist_total_time(template, vals):
        return vals['walk_total'] + vals['bike_total']

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    dists = [dist_walk_time_str, dist_walk_time, dist_walk_days_str, dist_walk_days,
             dist_walk_to_work, dist_walk_total, dist_bike_time_str, dist_bike_time,
             dist_bike_days, dist_bike_days_str, dist_bike_to_work, dist_bike_total,
             dist_total_time, dist_name]
    var_names = ['walk_time_str', 'walk_time', 'walk_days_str', 'walk_days', 'walk_to_work',
                 'walk_total', 'bike_time_str', 'bike_time', 'bike_days', 'bike_days_str',
                 'bike_to_work', 'bike_total', 'total_time', 'name']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_20(generator):
    marked_q = '''{name:s} rides his bike back and forth to work for each of his 5 workdays. His work is {work_distance:d} miles away. He also goes for a weekend bike ride of {weekend_ride:d} miles. If he can bike at {speed:d} mph how much time does he spend biking a week?'''
    marked_a = '''He bikes {work_distance:d}*2=<<{work_distance:d}*2={daily_work_miles:d}>>{daily_work_miles:d} miles each day for work.
So he bikes {daily_work_miles:d}*5=<<{daily_work_miles:d}*5={weekly_work_miles:d}>>{weekly_work_miles:d} miles for work.
That means he bikes a total of {weekly_work_miles:d}+{weekend_ride:d}=<<{weekly_work_miles:d}+{weekend_ride:d}={total_miles:d}>>{total_miles:d} miles for work.
So he bikes a total of {total_miles:d}/{speed:d}=<<{total_miles:d}/{speed:d}={total_hours:d}>>{total_hours:d} hours.
#### {total_hours:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names_male)

    def dist_speed(template, vals):
        return random.randint(5, 25)

    def dist_work_distance(template, vals):
        return random.randint(5, 30)

    def dist_daily_work_miles(template, vals):
        return vals['work_distance'] * 2

    def dist_weekly_work_miles(template, vals):
        return vals['daily_work_miles'] * 5

    def dist_total_hours(template, vals):
        return random.randint(5, 30)

    def dist_total_miles(template, vals):
        return vals['speed'] * vals['total_hours']

    def dist_weekend_ride(template, vals):
        return vals['total_miles'] - vals['weekly_work_miles']

    dists = [dist_name, dist_speed, dist_work_distance, dist_daily_work_miles,
             dist_weekly_work_miles, dist_total_hours, dist_total_miles, dist_weekend_ride]
    var_names = ['name', 'speed', 'work_distance', 'daily_work_miles', 'weekly_work_miles',
                 'total_hours', 'total_miles', 'weekend_ride']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_21(generator):
    marked_q = '''{name:s} bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. {name:s} bought {snowflake_stamps:d} snowflake stamps. She bought {more_truck:d} more truck stamps than snowflake stamps, and {less_rose:d} fewer rose stamps than truck stamps. How many stamps did {name:s} buy in all?'''
    marked_a = '''The number of truck stamps is {snowflake_stamps:d} + {more_truck:d} = <<{snowflake_stamps:d}+{more_truck:d}={truck_stamps:d}>>{truck_stamps:d}.
The number of rose stamps is {truck_stamps:d} - {less_rose:d} = <<{truck_stamps:d}-{less_rose:d}={rose_stamps:d}>>{rose_stamps:d}.
{name:s} bought {snowflake_stamps:d} + {truck_stamps:d} + {rose_stamps:d} = <<{snowflake_stamps:d}+{truck_stamps:d}+{rose_stamps:d}={total_stamps:d}>>{total_stamps:d} stamps in all.
#### {total_stamps:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names_female)

    def dist_snowflake_stamps(template, vals):
        return random.randint(2, 20)

    def dist_more_truck(template, vals):
        return random.randint(1, 20)

    def dist_truck_stamps(template, vals):
        return vals['snowflake_stamps'] + vals['more_truck']

    def dist_less_rose(template, vals):
        return random.randint(1, vals['truck_stamps'] - 1)

    def dist_rose_stamps(template, vals):
        return vals['truck_stamps'] - vals['less_rose']

    def dist_total_stamps(template, vals):
        return vals['snowflake_stamps'] + vals['truck_stamps'] + vals['rose_stamps']

    dists = [dist_name, dist_snowflake_stamps, dist_more_truck, dist_truck_stamps,
             dist_less_rose, dist_rose_stamps, dist_total_stamps]
    var_names = ['name', 'snowflake_stamps', 'more_truck', 'truck_stamps', 'less_rose',
                 'rose_stamps', 'total_stamps']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_22(generator):
    marked_q = '''Each bird eats {beetles_per_bird:d} beetles per day, each snake eats {birds_per_snake:d} birds per day, and each jaguar eats {snakes_per_jaguar:d} snakes per day. If there are {jaguars:d} jaguars in a forest, how many beetles are eaten each day?'''
    marked_a = '''First find the total number of snakes eaten: {snakes_per_jaguar:d} snakes/jaguar * {jaguars:d} jaguars = <<{snakes_per_jaguar:d}*{jaguars:d}={snakes_eaten:d}>>{snakes_eaten:d} snakes.
Then find the total number of birds eaten per day: {snakes_eaten:d} snakes * {birds_per_snake:d} birds/snake = <<{snakes_eaten:d}*{birds_per_snake:d}={birds_eaten:d}>>{birds_eaten:d} birds.
Then multiply the number of birds by the number of beetles per bird to find the total number of beetles eaten per day: {birds_eaten:d} birds * {beetles_per_bird:d} beetles/bird = <<{birds_eaten:d}*{beetles_per_bird:d}={beetles_eaten:d}>>{beetles_eaten:d} beetles.
#### {beetles_eaten:d}'''

    def dist_beetles_per_bird(template, vals):
        return random.randint(2, 50)

    def dist_birds_per_snake(template, vals):
        return random.randint(2, 10)

    def dist_snakes_per_jaguar(template, vals):
        return random.randint(2, 10)

    def dist_jaguars(template, vals):
        return random.randint(2, 20)

    def dist_snakes_eaten(template, vals):
        return vals['snakes_per_jaguar'] * vals['jaguars']

    def dist_birds_eaten(template, vals):
        return vals['snakes_eaten'] * vals['birds_per_snake']

    def dist_beetles_eaten(template, vals):
        return vals['birds_eaten'] * vals['beetles_per_bird']

    dists = [dist_beetles_per_bird, dist_birds_per_snake, dist_snakes_per_jaguar, dist_jaguars,
             dist_snakes_eaten, dist_birds_eaten, dist_beetles_eaten]
    var_names = ['beetles_per_bird', 'birds_per_snake', 'snakes_per_jaguar', 'jaguars',
                 'snakes_eaten', 'birds_eaten', 'beetles_eaten']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_23(generator):
    marked_q = '''{name1:s}’s last name has {less_letters:d} fewer letters than {name2:s}’s last name. If {name2:s} took {letters_removed:d} letters off her last name, she would have a last name {multiplier:s} the length of {name3:s}’s. {name3:s}’s full name is {name3:s} {last_name:s}. How many letters are in {name1:s}’s last name?'''
    marked_a = '''There are {last_name_length:d} letters in {name3:s}’s last name, so {name2:s}’s name is {last_name_length:d}*{factor:d} + {letters_removed:d} = <<{last_name_length:d}*{factor:d}+{letters_removed:d}={name2_last_name_length:d}>>{name2_last_name_length:d} letters long.
{name1:s}’s last name is {less_letters:d} letters shorter than {name2:s}’s, so there are {name2_last_name_length:d} - {less_letters:d} = <<{name2_last_name_length:d}-{less_letters:d}={name1_last_name_length:d}>>{name1_last_name_length:d} letters in {name1:s}’s last name.
#### {name1_last_name_length:d}'''

    def dist_name1(template, vals):
        return random.choice(generator.names_female)

    def dist_name2(template, vals):
        available_names = [name for name in generator.names_female if name != vals['name1']]
        return random.choice(available_names)

    def dist_name3(template, vals):
        available_names = [name for name in generator.names_female if name not in [vals['name1'], vals['name2']]]
        return random.choice(available_names)

    def dist_last_name(template, vals):
        options = ["Ngo", "Adam", "Black", "Austin", "Zickert", "Smithson", "Christoph"]
        return random.choice(options)

    def dist_last_name_length(template, vals):
        return {"Ngo": 3, "Adam": 4, "Black": 5, "Austin": 6, "Zickert": 7, "Smithson": 8, "Christoph": 9}[vals['last_name']]

    def dist_letters_removed(template, vals):
        return random.randint(2, 10)

    def dist_multiplier(template, vals):
        return random.choice(["twice", "three times", "four times"])

    def dist_factor(template, vals):
        return {"twice": 2, "three times": 3, "four times": 4}[vals['multiplier']]

    def dist_name2_last_name_length(template, vals):
        return vals['last_name_length'] * vals['factor'] + vals['letters_removed']

    def dist_less_letters(template, vals):
        return random.randint(2, vals['name2_last_name_length'] - 2)

    def dist_name1_last_name_length(template, vals):
        return vals['name2_last_name_length'] - vals['less_letters']

    dists = [dist_name1, dist_name2, dist_name3, dist_last_name, dist_last_name_length,
             dist_letters_removed, dist_multiplier, dist_factor, dist_name2_last_name_length,
             dist_less_letters, dist_name1_last_name_length]
    var_names = ['name1', 'name2', 'name3', 'last_name', 'last_name_length', 'letters_removed',
                 'multiplier', 'factor', 'name2_last_name_length', 'less_letters', 'name1_last_name_length']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_24(generator):
    marked_q = '''{name:s}'s favorite store was having a summer clearance. For ${total_spent:d} she bought {shorts_pairs:d} pairs of shorts for ${shorts_price:d} each and {shoes_pairs:d} pairs of shoes for ${shoes_price:d} each. She also bought {tops:d} tops, all at the same price. How much did each top cost?'''
    marked_a = '''She bought {shorts_pairs:d} shorts at ${shorts_price:d} each so {shorts_pairs:d}*{shorts_price:d}=$<<{shorts_pairs:d}*{shorts_price:d}={shorts_cost:d}>>{shorts_cost:d}.
She bought {shoes_pairs:d} pair of shoes at ${shoes_price:d} each so {shoes_pairs:d}*{shoes_price:d}=$<<{shoes_pairs:d}*{shoes_price:d}={shoes_cost:d}>>{shoes_cost:d}.
The shorts and shoes cost her {shorts_cost:d}+{shoes_cost:d} = $<<{shorts_cost:d}+{shoes_cost:d}={shorts_shoes_cost:d}>>{shorts_shoes_cost:d}.
We know she spent {total_spent:d} total and the shorts and shoes cost ${shorts_shoes_cost:d} which left a difference of {total_spent:d}-{shorts_shoes_cost:d} = $<<{total_spent:d}-{shorts_shoes_cost:d}={tops_total:d}>>{tops_total:d}.
She bought {tops:d} tops for a total of ${tops_total:d} so {tops_total:d}/{tops:d} = ${top_price:d}.
#### {top_price:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names_female)

    def dist_shorts_pairs(template, vals):
        return random.randint(2, 10)

    def dist_shorts_price(template, vals):
        return random.randint(5, 20)

    def dist_shorts_cost(template, vals):
        return vals['shorts_pairs'] * vals['shorts_price']

    def dist_shoes_pairs(template, vals):
        return random.randint(2, 10)

    def dist_shoes_price(template, vals):
        return random.randint(5, 30)

    def dist_shoes_cost(template, vals):
        return vals['shoes_pairs'] * vals['shoes_price']

    def dist_shorts_shoes_cost(template, vals):
        return vals['shorts_cost'] + vals['shoes_cost']

    def dist_tops(template, vals):
        return random.randint(2, 10)

    def dist_top_price(template, vals):
        return random.randint(5, 20)

    def dist_tops_total(template, vals):
        return vals['tops'] * vals['top_price']

    def dist_total_spent(template, vals):
        return vals['shorts_shoes_cost'] + vals['tops_total']

    dists = [dist_name, dist_shorts_pairs, dist_shorts_price, dist_shorts_cost, dist_shoes_pairs,
             dist_shoes_price, dist_shoes_cost, dist_shorts_shoes_cost, dist_tops, dist_top_price,
             dist_tops_total, dist_total_spent]
    var_names = ['name', 'shorts_pairs', 'shorts_price', 'shorts_cost', 'shoes_pairs',
                 'shoes_price', 'shoes_cost', 'shorts_shoes_cost', 'tops', 'top_price',
                 'tops_total', 'total_spent']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_25(generator):
    marked_q = '''{name:s} does her grocery shopping on Saturday. She does her shopping only at a specific store where she is allowed a credit of ${credit:d}, which must be paid in full before her next shopping trip. That week she spent the full credit limit and paid ${payment1:d} of it on Tuesday and ${payment2:d} of it on Thursday. How much credit will {name:s} need to pay before her next shopping trip?'''
    marked_a = '''So far, {name:s} has paid back ${payment1:d}+${payment2:d}=$<<{payment1:d}+{payment2:d}={total_paid:d}>>{total_paid:d} of the credit.
So she still needs to pay ${credit:d}-${total_paid:d}=$<<{credit:d}-{total_paid:d}={remaining_credit:d}>>{remaining_credit:d}.
#### {remaining_credit:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names_female)

    def dist_payment1(template, vals):
        return random.randint(1, 100)

    def dist_payment2(template, vals):
        return random.randint(1, 100)

    def dist_total_paid(template, vals):
        return vals['payment1'] + vals['payment2']

    def dist_credit(template, vals):
        return random.randint((vals['total_paid'] // 10 + 1) * 10, 50 * 10)

    def dist_remaining_credit(template, vals):
        return vals['credit'] - vals['total_paid']

    dists = [dist_name, dist_payment1, dist_payment2, dist_total_paid, dist_credit, dist_remaining_credit]
    var_names = ['name', 'payment1', 'payment2', 'total_paid', 'credit', 'remaining_credit']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_26(generator):
    marked_q = '''{name:s} earned ${a:d} working odd jobs around the neighborhood. 
She spent a {b:d}th of it on a milkshake and put a {c:d}th of the rest in her savings account.
She left the remaining money in her wallet. 
Her dog got ahold of her wallet and shredded all the money inside but ${d:d}.
How many dollars did {name:s} lose?'''
    marked_a = '''{name:s} spent {a:d} / {b:d} = $<<{a:d}/{b:d}={milkshake:d}>>{milkshake:d} on a milkshake.
She had {a:d} - {milkshake:d} = $<<{a:d}-{milkshake:d}={remaining_after_milkshake:d}>>{remaining_after_milkshake:d} left.
She put {remaining_after_milkshake:d} / {c:d} = $<<{remaining_after_milkshake:d}/{c:d}={savings:d}>>{savings:d} in her savings account.
She left {remaining_after_milkshake:d} - {savings:d} = $<<{remaining_after_milkshake:d}-{savings:d}={wallet:d}>>{wallet:d} in her wallet.
Her dog shredded all the money in her wallet but ${d:d}, so {name:s} lost {wallet:d} - {d:d} = $<<{wallet:d}-{d:d}={lost:d}>>{lost:d}.
#### {lost:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(20, 100)

    def dist_b(template, vals):
        a = vals['a']
        divisors = [i for i in range(2, a + 1) if a % i == 0]
        return random.choice(divisors)

    def dist_milkshake(template, vals):
        return vals['a'] // vals['b']

    def dist_remaining_after_milkshake(template, vals):
        return vals['a'] - vals['milkshake']

    def dist_c(template, vals):
        remaining = vals['remaining_after_milkshake']
        divisors = [j for j in range(2, remaining + 1) if remaining % j == 0]
        return random.choice(divisors)

    def dist_savings(template, vals):
        return vals['remaining_after_milkshake'] // vals['c']

    def dist_wallet(template, vals):
        return vals['remaining_after_milkshake'] - vals['savings']

    def dist_d(template, vals):
        wallet = vals['wallet']
        return random.randint(5, wallet)

    def dist_lost(template, vals):
        return vals['wallet'] - vals['d']

    dists = [dist_name, dist_a, dist_b, dist_milkshake, dist_remaining_after_milkshake, dist_c, dist_savings, dist_wallet, dist_d, dist_lost]
    var_names = ['name', 'a', 'b', 'milkshake', 'remaining_after_milkshake', 'c', 'savings', 'wallet', 'd', 'lost']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_27(generator):
    marked_q = '''There are {a:d} roses in a garden. 
There are {b:d} tulips. 
There are {c:d} daisies. 
What percentage of flowers are not roses?'''
    marked_a = '''There are {a:d}+{b:d}+{c:d}=<<{a:d}+{b:d}+{c:d}={total:d}>>{total:d} flowers total.
There are {b:d}+{c:d}=<<{b:d}+{c:d}={not_roses:d}>>{not_roses:d} flowers that are not roses.
Therefore, ({not_roses:d}/{total:d})*100=<<({not_roses:d}/{total:d})*100={percentage:d}>>{percentage:d}% of the flowers are not roses.
#### {percentage:d}'''

    def dist_b(template, vals):
        return random.randint(5, 100)

    def dist_c(template, vals):
        return random.randint(5, 100)

    def dist_a(template, vals):
        dividend = vals['b'] + vals['c']
        possible_a = [i for i in range(1, dividend + 1) if ((dividend / (i + dividend)) * 100) % 1 == 0]
        if not possible_a:
            raise ValueError("No suitable a found for b={} and c={}".format(vals['b'], vals['c']))
        return random.choice(possible_a)

    def dist_total(template, vals):
        return vals['a'] + vals['b'] + vals['c']

    def dist_not_roses(template, vals):
        return vals['b'] + vals['c']

    def dist_percentage(template, vals):
        return int((vals['not_roses'] / vals['total']) * 100)

    dists = [dist_b, dist_c, dist_a, dist_total, dist_not_roses, dist_percentage]
    var_names = ['b', 'c', 'a', 'total', 'not_roses', 'percentage']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_28(generator):
    marked_q = '''{name:s}'s assignment was divided into three parts. 
He finished the first part of his assignment in {a:d} minutes. 
It took him twice as long to finish the second part. 
If he was able to finish his assignment in {b_hours:d} hours, how many minutes did it take {name:s} to finish the third part of the assignment?'''
    marked_a = '''It took {name:s} {a:d} x 2 = <<{a:d}*2={second_part:d}>>{second_part:d} minutes to finish the second part of the assignment.
{name:s} finished the first and second parts of the assignment in {a:d} + {second_part:d} = <<{a:d}+{second_part:d}={first_second:d}>>{first_second:d} minutes.
He finished the entire assignment in 60 x {b_hours:d} = <<60*{b_hours:d}={total_minutes:d}>>{total_minutes:d} minutes.
Therefore, it took {name:s} {total_minutes:d} - {first_second:d} = <<{total_minutes:d}-{first_second:d}={third_part:d}>>{third_part:d} minutes to finish the third part of the assignment.
#### {third_part:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_b_hours(template, vals):
        return random.randint(2, 10)

    def dist_a(template, vals):
        b_hours = vals['b_hours']
        max_a = (b_hours * 60) // 3 - 1  # Ensure third_part > 0
        return random.randint(5, max_a)

    def dist_second_part(template, vals):
        return 2 * vals['a']

    def dist_first_second(template, vals):
        return vals['a'] + vals['second_part']

    def dist_total_minutes(template, vals):
        return vals['b_hours'] * 60

    def dist_third_part(template, vals):
        return vals['total_minutes'] - vals['first_second']

    dists = [dist_name, dist_b_hours, dist_a, dist_second_part, dist_first_second, dist_total_minutes, dist_third_part]
    var_names = ['name', 'b_hours', 'a', 'second_part', 'first_second', 'total_minutes', 'third_part']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_29(generator):
    marked_q = '''{name:s} bought {a:d} kilograms of butter to make cookies. 
She used 1/{b:d}th of it for chocolate chip cookies, 1/{c:d}th of it for peanut butter cookies, and 1/{d:d}th of the remaining butter for sugar cookies. 
How many kilograms of butter are left after making those three kinds of cookies?'''
    marked_a = '''{name:s} used {a:d}/{b:d} = <<{a:d}/{b:d}={chocolate:d}>>{chocolate:d} kilograms of butter for the chocolate chip cookies.
Then, she used {a:d}/{c:d} = <<{a:d}/{c:d}={peanut:d}>>{peanut:d} kilograms of butter for the peanut butter cookies.
She used {chocolate:d} + {peanut:d} = <<{chocolate:d}+{peanut:d}={used_so_far:d}>>{used_so_far:d} kilograms of butter for the chocolate and peanut butter cookies.
So, only {a:d} - {used_so_far:d} = <<{a:d}-{used_so_far:d}={remaining_after_two:d}>>{remaining_after_two:d} kilograms of butter was left.
Then, {name:s} used {remaining_after_two:d}/{d:d} = <<{remaining_after_two:d}/{d:d}={sugar:d}>>{sugar:d} kilograms of butter for the sugar cookies.
Therefore, only {remaining_after_two:d} - {sugar:d} = <<{remaining_after_two:d}-{sugar:d}={left:d}>>{left:d} kilograms of butter were left.
#### {left:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(10, 100)

    def dist_b(template, vals):
        a = vals['a']
        divisors = [i for i in range(3, a + 1) if a % i == 0]
        return random.choice(divisors)

    def dist_c(template, vals):
        a = vals['a']
        divisors = [i for i in range(3, a + 1) if a % i == 0]
        return random.choice(divisors)

    def dist_chocolate(template, vals):
        return vals['a'] // vals['b']

    def dist_peanut(template, vals):
        return vals['a'] // vals['c']

    def dist_used_so_far(template, vals):
        return vals['chocolate'] + vals['peanut']

    def dist_remaining_after_two(template, vals):
        return vals['a'] - vals['used_so_far']

    def dist_d(template, vals):
        remaining = vals['remaining_after_two']
        divisors = [j for j in range(2, remaining + 1) if remaining % j == 0]
        return random.choice(divisors)

    def dist_sugar(template, vals):
        return vals['remaining_after_two'] // vals['d']

    def dist_left(template, vals):
        return vals['remaining_after_two'] - vals['sugar']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_chocolate, dist_peanut, dist_used_so_far, dist_remaining_after_two, dist_d, dist_sugar, dist_left]
    var_names = ['name', 'a', 'b', 'c', 'chocolate', 'peanut', 'used_so_far', 'remaining_after_two', 'd', 'sugar', 'left']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_30(generator):
    marked_q = '''A {subject:s} student wants to find out the average daily allowance of the middle school students. 
According to his survey, {a:d}/{b:d} of the students receive an average of ${c:d} allowance per day while the rest gets an average of ${d:d} a day. 
If he surveyed {e:d} students, what is the total amount of money those {e:d} students get in a day?'''
    marked_a = '''There are {e:d} students x {a:d}/{b:d} = <<{e:d}*{a:d}/{b:d}={group1:d}>>{group1:d} students who have a ${c:d} daily allowance.
While there are {e:d} students - {group1:d} students = <<{e:d}-{group1:d}={group2:d}>>{group2:d} students who have a ${d:d} daily allowance.
The sum of the allowances of the {group1:d} students who received ${c:d} daily is {group1:d} students x ${c:d}/day = $<<{group1:d}*{c:d}={sum_group1:d}>>{sum_group1:d}.
The sum of the allowances of the {group2:d} students who received ${d:d} daily is {group2:d} students x ${d:d}/day = $<<{group2:d}*{d:d}={sum_group2:d}>>{sum_group2:d}.
The total daily amount of money of those {e:d} students is ${sum_group1:d} + ${sum_group2:d} = $<<{sum_group1:d}+{sum_group2:d}={total:d}>>{total:d}.
#### {total:d}'''

    def dist_subject(template, vals):
        return random.choice(['Mathematics', 'English', 'Business'])

    def dist_a(template, vals):
        return random.randint(1, 5)

    def dist_b(template, vals):
        a = vals['a']
        dividend = a * random.randint(5, 100)
        divisors = [i for i in range(a + 1, dividend + 1) if dividend % i == 0]
        return random.choice(divisors)

    def dist_c(template, vals):
        return random.randint(5, 20)

    def dist_d(template, vals):
        return random.randint(5, 20)

    def dist_e(template, vals):
        return random.randint(5, 100)

    def dist_group1(template, vals):
        return int(vals['e'] * (vals['a'] / vals['b']))

    def dist_group2(template, vals):
        return vals['e'] - vals['group1']

    def dist_sum_group1(template, vals):
        return vals['group1'] * vals['c']

    def dist_sum_group2(template, vals):
        return vals['group2'] * vals['d']

    def dist_total(template, vals):
        return vals['sum_group1'] + vals['sum_group2']

    dists = [dist_subject, dist_a, dist_b, dist_c, dist_d, dist_e, dist_group1, dist_group2, dist_sum_group1, dist_sum_group2, dist_total]
    var_names = ['subject', 'a', 'b', 'c', 'd', 'e', 'group1', 'group2', 'sum_group1', 'sum_group2', 'total']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_31(generator):
    marked_q = '''Every hour {name:s} has to collect the coins out of the fountain inside the mall. 
During the first hour, she collected {a:d} coins. 
For the next two hours, she collected {b:d} coins from the fountain. 
In the fourth hour, she collected {c:d} coins from the fountain but she gave {d:d} of them to her coworker so she could buy a soda. 
How many coins did she have after the fourth hour?'''
    marked_a = '''{a:d} coins collected in hour one
{b:d} coins collected in hour two
{b:d} coins collected in hour three
{c:d} coins collected in hour four
Before giving her coworker some coins there were {a:d}+{b:d}+{b:d}+{c:d}=<<{a:d}+{b:d}+{b:d}+{c:d}={total_before:d}>>{total_before:d} coins
The number of coins after giving {d:d} to her coworker is {total_before:d}-{d:d}=<<{total_before:d}-{d:d}={final_coins:d}>>{final_coins:d}
#### {final_coins:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 100)

    def dist_b(template, vals):
        return random.randint(5, 100)

    def dist_c(template, vals):
        return random.randint(5, 100)

    def dist_total_before(template, vals):
        return vals['a'] + 2 * vals['b'] + vals['c']

    def dist_d(template, vals):
        total_collected = vals['total_before']
        return random.randint(5, total_collected)

    def dist_final_coins(template, vals):
        return vals['total_before'] - vals['d']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_total_before, dist_d, dist_final_coins]
    var_names = ['name', 'a', 'b', 'c', 'total_before', 'd', 'final_coins']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_32(generator):
    marked_q = '''{name:s}'s two daughters play softball on different teams. 
They each have {a:d} games this season. Each team practices {b:d} hours for every game they play. 
If each game lasts for {c:d} hours, how many hours will {name:s} spend at the field watching his daughters play and practice altogether?'''
    marked_a = '''{name:s} will spend {a:d} games x {c:d} hours per game = <<{a:d}*{c:d}={game_time_per_daughter:d}>>{game_time_per_daughter:d} hours watching one daughter play her games.
He will spend {game_time_per_daughter:d} x 2 = <<{game_time_per_daughter:d}*2={total_game_time:d}>>{total_game_time:d} hours watching both daughters play their games.
He will spend {a:d} games x {b:d} hours of practice = <<{a:d}*{b:d}={practice_time_per_daughter:d}>>{practice_time_per_daughter:d} hours watching one daughter practice.
He will spend {practice_time_per_daughter:d} x 2 = <<{practice_time_per_daughter:d}*2={total_practice_time:d}>>{total_practice_time:d} hours watching both daughters practice.
He will spend a total of {total_game_time:d} hours watching games + {total_practice_time:d} hours watching practice = <<{total_game_time:d}+{total_practice_time:d}={total_time:d}>>{total_time:d} hours.
#### {total_time:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 20)

    def dist_b(template, vals):
        return random.randint(1, 10)

    def dist_c(template, vals):
        return random.randint(1, 10)

    def dist_game_time_per_daughter(template, vals):
        return vals['a'] * vals['c']

    def dist_total_game_time(template, vals):
        return vals['game_time_per_daughter'] * 2

    def dist_practice_time_per_daughter(template, vals):
        return vals['a'] * vals['b']

    def dist_total_practice_time(template, vals):
        return vals['practice_time_per_daughter'] * 2

    def dist_total_time(template, vals):
        return vals['total_game_time'] + vals['total_practice_time']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_game_time_per_daughter, dist_total_game_time, dist_practice_time_per_daughter, dist_total_practice_time, dist_total_time]
    var_names = ['name', 'a', 'b', 'c', 'game_time_per_daughter', 'total_game_time', 'practice_time_per_daughter', 'total_practice_time', 'total_time']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_33(generator):
    marked_q = '''A bear is preparing to hibernate for the winter and needs to gain {c:d} pounds. 
At the end of summer, the bear feasts on berries and small woodland animals. 
During autumn, it devours acorns and salmon. 
It gained a {b:d}th of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns.
Salmon made up {a:d}th of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?'''
    marked_a = '''The bear gained 1 / {b:d} * {c:d} = <<1/{b:d}*{c:d}={berries:d}>>{berries:d} pounds from berries.
It gained 2 * {berries:d} = <<2*{berries:d}={acorns:d}>>{acorns:d} pounds from acorns.
It still needed {c:d} - {berries:d} - {acorns:d} = <<{c:d}-{berries:d}-{acorns:d}={remaining_after_acorns:d}>>{remaining_after_acorns:d} pounds.
Thus, it gained {remaining_after_acorns:d} / {a:d} = <<{remaining_after_acorns:d}/{a:d}={salmon:d}>>{salmon:d} pounds from salmon.
Therefore, the bear gained {remaining_after_acorns:d} - {salmon:d} = <<{remaining_after_acorns:d}-{salmon:d}={small_animals:d}>>{small_animals:d} pounds from small animals.
#### {small_animals:d}'''

    def dist_a(template, vals):
        return random.randint(2, 10)

    def dist_b(template, vals):
        return random.randint(2, 10)

    def dist_c(template, vals):
        a = vals['a']
        b = vals['b']
        c = random.randint(100, 1000)
        while c % b != 0 or (c - c//b - 2*(c//b)) % a != 0:
            c += 1
        return c

    def dist_berries(template, vals):
        return vals['c'] // vals['b']

    def dist_acorns(template, vals):
        return 2 * vals['berries']

    def dist_remaining_after_acorns(template, vals):
        return vals['c'] - vals['berries'] - vals['acorns']

    def dist_salmon(template, vals):
        return vals['remaining_after_acorns'] // vals['a']

    def dist_small_animals(template, vals):
        return vals['remaining_after_acorns'] - vals['salmon']

    dists = [dist_a, dist_b, dist_c, dist_berries, dist_acorns, dist_remaining_after_acorns, dist_salmon, dist_small_animals]
    var_names = ['a', 'b', 'c', 'berries', 'acorns', 'remaining_after_acorns', 'salmon', 'small_animals']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_34(generator):
    marked_q = '''There are {c:d} liters of oil in {b:d} cans. 
If {a:d} of the cans are holding {d:d} liters each, how much oil is each of the remaining cans holding?'''
    marked_a = '''{a:d} cans are holding {d:d} liters each for a total of {a:d} * {d:d} = <<{a:d}*{d:d}={total_known:d}>>{total_known:d} liters.
There are {c:d} - {total_known:d} = <<{c:d}-{total_known:d}={remaining_oil:d}>>{remaining_oil:d} liters left.
There are {b:d} - {a:d} = <<{b:d}-{a:d}={remaining_cans:d}>>{remaining_cans:d} cans left.
Each of the remaining cans is holding {remaining_oil:d} / {remaining_cans:d} = <<{remaining_oil:d}/{remaining_cans:d}={oil_per_can:d}>>{oil_per_can:d} liters each.
#### {oil_per_can:d}'''

    def dist_a(template, vals):
        return random.randint(10, 50)

    def dist_d(template, vals):
        return random.randint(5, 10)

    def dist_b(template, vals):
        a = vals['a']
        d = vals['d']
        possible_b = [i for i in range(a + 1, 100) if (i - a) > 0]
        return random.choice(possible_b)

    def dist_c(template, vals):
        a = vals['a']
        b = vals['b']
        d = vals['d']
        oil_per_can = random.randint(1, 100)
        remaining_cans = b - a
        total_known = a * d
        remaining_oil = oil_per_can * remaining_cans
        return total_known + remaining_oil

    def dist_total_known(template, vals):
        return vals['a'] * vals['d']

    def dist_remaining_oil(template, vals):
        return vals['c'] - vals['total_known']

    def dist_remaining_cans(template, vals):
        return vals['b'] - vals['a']

    def dist_oil_per_can(template, vals):
        return vals['remaining_oil'] // vals['remaining_cans']

    dists = [dist_a, dist_d, dist_b, dist_c, dist_total_known, dist_remaining_oil, dist_remaining_cans, dist_oil_per_can]
    var_names = ['a', 'd', 'b', 'c', 'total_known', 'remaining_oil', 'remaining_cans', 'oil_per_can']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_35(generator):
    marked_q = '''{name:s}'s workout goal is {a:d} situps. 
On Monday, {name:s} was only able to do {b:d} situps, so she decided that she would make up for the rest on Tuesday. 
However, she was only able to do {c:d} situps on Tuesday. 
How many situps would {name:s} have to do on Wednesday to meet her minimum goal and make up for the ones she didn't do?'''
    marked_a = '''On Monday, {name:s} was short of {a:d} - {b:d} = <<{a:d}-{b:d}={short_monday:d}>>{short_monday:d} situps.
On Tuesday, {name:s} was short of {a:d} - {c:d} = <<{a:d}-{c:d}={short_tuesday:d}>>{short_tuesday:d} situps.
On Wednesday, {name:s} would have to do {a:d} + {short_monday:d} + {short_tuesday:d} = <<{a:d}+{short_monday:d}+{short_tuesday:d}={total_wednesday:d}>>{total_wednesday:d} situps.
#### {total_wednesday:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(50, 100)

    def dist_b(template, vals):
        return random.randint(5, vals['a'] - 1)

    def dist_c(template, vals):
        return random.randint(5, vals['a'] - 1)

    def dist_short_monday(template, vals):
        return vals['a'] - vals['b']

    def dist_short_tuesday(template, vals):
        return vals['a'] - vals['c']

    def dist_total_wednesday(template, vals):
        return vals['a'] + vals['short_monday'] + vals['short_tuesday']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_short_monday, dist_short_tuesday, dist_total_wednesday]
    var_names = ['name', 'a', 'b', 'c', 'short_monday', 'short_tuesday', 'total_wednesday']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_36(generator):
    marked_q = '''{name:s} earns ${a:d} an hour while working at his main job.  
He earns {b:d}% less while working his second job.  
He works {c:d} hours at his main job and half that much at his second job.  
How much does he earn per week?'''
    marked_a = '''{name:s} earns {a:d} * {b:d}/100 = $<<{a:d}*{b:d}/100={less_amount:d}>>{less_amount:d} less while working his second job.
So he earns {a:d} - {less_amount:d} = $<<{a:d}-{less_amount:d}={second_job_rate:d}>>{second_job_rate:d} an hour.
At his first job he earns {a:d} * {c:d} = $<<{a:d}*{c:d}={main_job_earnings:d}>>{main_job_earnings:d}.
He works {c:d}/2 = <<{c:d}/2={second_job_hours:d}>>{second_job_hours:d} hours at his second job.
So he earns {second_job_hours:d} * {second_job_rate:d} = $<<{second_job_hours:d}*{second_job_rate:d}={second_job_earnings:d}>>{second_job_earnings:d}.
So he earns {main_job_earnings:d} + {second_job_earnings:d} = $<<{main_job_earnings:d}+{second_job_earnings:d}={total_earnings:d}>>{total_earnings:d} a week.
#### {total_earnings:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 25)

    def dist_b(template, vals):
        return random.randint(5, 50)

    def dist_c(template, vals):
        return random.randint(10, 50) * 2  # Ensure even for half hours

    def dist_less_amount(template, vals):
        return (vals['a'] * vals['b']) // 100

    def dist_second_job_rate(template, vals):
        return vals['a'] - vals['less_amount']

    def dist_main_job_earnings(template, vals):
        return vals['a'] * vals['c']

    def dist_second_job_hours(template, vals):
        return vals['c'] // 2

    def dist_second_job_earnings(template, vals):
        return vals['second_job_hours'] * vals['second_job_rate']

    def dist_total_earnings(template, vals):
        return vals['main_job_earnings'] + vals['second_job_earnings']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_less_amount, dist_second_job_rate, dist_main_job_earnings, dist_second_job_hours, dist_second_job_earnings, dist_total_earnings]
    var_names = ['name', 'a', 'b', 'c', 'less_amount', 'second_job_rate', 'main_job_earnings', 'second_job_hours', 'second_job_earnings', 'total_earnings']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_37(generator):
    marked_q = '''{name:s} mows one lawn and charges ${a:d}. 
Last week he mowed {b:d} lawns and {d:d} customers each gave him a ${c:d} tip. 
How many dollars did {name:s} earn mowing lawns last week?'''
    marked_a = '''{a:d} * {b:d} = $<<{a:d}*{b:d}={earnings_from_mowing:d}>>{earnings_from_mowing:d}
{d:d} * {c:d} = $<<{d:d}*{c:d}={earnings_from_tips:d}>>{earnings_from_tips:d}
{earnings_from_mowing:d} + {earnings_from_tips:d} = $<<{earnings_from_mowing:d}+{earnings_from_tips:d}={total_earnings:d}>>{total_earnings:d}
{name:s} earned ${total_earnings:d} mowing lawns last week.
#### {total_earnings:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 50)

    def dist_b(template, vals):
        return random.randint(5, 25)

    def dist_c(template, vals):
        return random.randint(5, 25)

    def dist_d(template, vals):
        return random.randint(1, vals['b'])

    def dist_earnings_from_mowing(template, vals):
        return vals['a'] * vals['b']

    def dist_earnings_from_tips(template, vals):
        return vals['d'] * vals['c']

    def dist_total_earnings(template, vals):
        return vals['earnings_from_mowing'] + vals['earnings_from_tips']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_d, dist_earnings_from_mowing, dist_earnings_from_tips, dist_total_earnings]
    var_names = ['name', 'a', 'b', 'c', 'd', 'earnings_from_mowing', 'earnings_from_tips', 'total_earnings']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_38(generator):
    marked_q = '''{name:s} has been planning to buy a laptop which costs ${a:d}. 
A computer shop accepts payment in installments of ${b:d} per month provided that a {c:d}% down payment is made.
If {name:s} wants to pay an additional ${d:d} for the down payment, how much will her balance be after paying for {e:d} months?'''
    marked_a = '''{name:s} has to make a ${a:d} x {c:d}/100 = $<<{a:d}*{c:d}/100={down_payment:d}>>{down_payment:d} down payment.
Since {name:s} wants to pay ${d:d} more for the down payment, her total down payment will be ${down_payment:d} + ${d:d} = $<<{down_payment:d}+{d:d}={total_down_payment:d}>>{total_down_payment:d}.
So her remaining balance payable over a year is ${a:d} - ${total_down_payment:d} = $<<{a:d}-{total_down_payment:d}={balance:d}>>{balance:d}.
{name:s} has to make a monthly payment of ${balance:d}/12 = $<<{balance:d}/12={monthly_payment:d}>>{monthly_payment:d}/month.
The total cost of her payments for {e:d} months is ${monthly_payment:d} * {e:d} = $<<{monthly_payment:d}*{e:d}={total_payments:d}>>{total_payments:d}.
Therefore, {name:s}'s balance after {e:d} months is ${balance:d} - ${total_payments:d} = $<<{balance:d}-{total_payments:d}={remaining_balance:d}>>{remaining_balance:d}.
#### {remaining_balance:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(500, 2000)

    def dist_b(template, vals):
        return random.randint(50, 100)

    def dist_c(template, vals):
        return random.randint(10, 50)

    def dist_d(template, vals):
        return random.randint(5, 50)

    def dist_e(template, vals):
        return random.randint(1, 12)

    def dist_down_payment(template, vals):
        return (vals['a'] * vals['c']) // 100

    def dist_total_down_payment(template, vals):
        return vals['down_payment'] + vals['d']

    def dist_balance(template, vals):
        return vals['a'] - vals['total_down_payment']

    def dist_monthly_payment(template, vals):
        return vals['balance'] // 12

    def dist_total_payments(template, vals):
        return vals['monthly_payment'] * vals['e']

    def dist_remaining_balance(template, vals):
        return vals['balance'] - vals['total_payments']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_d, dist_e, dist_down_payment, dist_total_down_payment, dist_balance, dist_monthly_payment, dist_total_payments, dist_remaining_balance]
    var_names = ['name', 'a', 'b', 'c', 'd', 'e', 'down_payment', 'total_down_payment', 'balance', 'monthly_payment', 'total_payments', 'remaining_balance']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

# def template_39(generator):
#     marked_q = '''{name:s} and Mia are competing in a week long race. 
# They have one week to run {c:d} miles. On the first three days {name:s} averages {a:d} miles a day.
# On day four she runs {b:d} miles. Mia averages {d:d} miles a day over the first 4 days. 
# What is the average of their average that they have to run over the final three days?'''
#     marked_a = '''{name:s} runs 3 x {a:d} = <<3*{a:d}={first_three_days:d}>>{first_three_days:d} miles in the first three days.
# {name:s} has {c:d} - {b:d} - {first_three_days:d} = <<{c:d}-{b:d}-{first_three_days:d}={jesse_remaining:d}>>{jesse_remaining:d} miles left to run.
# {name:s} has to run an average of {jesse_remaining:d} / 3 = <<{jesse_remaining:d}/3={jesse_daily_average:d}>>{jesse_daily_average:d} miles a day.
# Mia runs 4 x {d:d} = <<4*{d:d}={mia_first_four:d}>>{mia_first_four:d} miles over the first four days.
# She has {c:d} - {mia_first_four:d} = <<{c:d}-{mia_first_four:d}={mia_remaining:d}>>{mia_remaining:d} miles left to run.
# She has to run {mia_remaining:d} / 3 = <<{mia_remaining:d}/3={mia_daily_average:d}>>{mia_daily_average:d} miles a day.
# The total they both have to run is {jesse_daily_average:d} + {mia_daily_average:d} = <<{jesse_daily_average:d}+{mia_daily_average:d}={total_daily:d}>>{total_daily:d} miles a day.
# The average they have to run per day on average is {total_daily:d} / 2 = <<{total_daily:d}/2={average_daily:d}>>{average_daily:d} miles.
# #### {average_daily:d}'''

#     def dist_name(template, vals):
#         return random.choice(generator.names)

#     def dist_a(template, vals):
#         return random.randint(2, 10)

#     def dist_b(template, vals):
#         return random.randint(5, 10)

#     def dist_d(template, vals):
#         return random.randint(2, 10)

#     def dist_c(template, vals):
#         a = vals['a']
#         b = vals['b']
#         d = vals['d']
#         min_c = max(3 * a + b, 4 * d) + 3
#         c = random.randint(min_c, 500)
#         while (c - b - 3 * a) % 3 != 0 or (c - 4 * d) % 3 != 0:
#             c += 1
#         return c

#     def dist_first_three_days(template, vals):
#         return 3 * vals['a']

#     def dist_jesse_remaining(template, vals):
#         return vals['c'] - vals['b'] - vals['first_three_days']

#     def dist_jesse_daily_average(template, vals):
#         return vals['jesse_remaining'] // 3

#     def dist_mia_first_four(template, vals):
#         return 4 * vals['d']

#     def dist_mia_remaining(template, vals):
#         return vals['c'] - vals['mia_first_four']

#     def dist_mia_daily_average(template, vals):
#         return vals['mia_remaining'] // 3

#     def dist_total_daily(template, vals):
#         return vals['jesse_daily_average'] + vals['mia_daily_average']

#     def dist_average_daily(template, vals):
#         return vals['total_daily'] // 2

#     dists = [dist_name, dist_a, dist_b, dist_d, dist_c, dist_first_three_days, dist_jesse_remaining, dist_jesse_daily_average, dist_mia_first_four, dist_mia_remaining, dist_mia_daily_average, dist_total_daily, dist_average_daily]
#     var_names = ['name', 'a', 'b', 'd', 'c', 'first_three_days', 'jesse_remaining', 'jesse_daily_average', 'mia_first_four', 'mia_remaining', 'mia_daily_average', 'total_daily', 'average_daily']
#     return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_40(generator):
    marked_q = '''The ratio of coins that Elsa has to that which {name:s} has is {a:d}:{b:d}. 
If the total number of coins they have is {c:d}, and {name:s} spends 3/4 of what she has on toys, how many will she remain with?'''
    marked_a = '''The total ratio of the coins they both have is {a:d}+{b:d} = <<{a:d}+{b:d}={total_ratio:d}>>{total_ratio:d}.
The fraction of the ratio representing the number of coins that {name:s} has is {b:d}/{total_ratio:d}, and since the total number of coins they both have is {c:d}, {name:s} has {b:d}/{total_ratio:d}*{c:d} = <<{b:d}/{total_ratio:d}*{c:d}={name_coins:d}>>{name_coins:d} coins.
When {name:s} spends 3/4 of what she has, she parts with 3/4*{name_coins:d} = <<3/4*{name_coins:d}={spent:d}>>{spent:d} coins.
She still has {name_coins:d} coins - {spent:d} coins = <<{name_coins:d}-{spent:d}={remaining:d}>>{remaining:d} coins.
#### {remaining:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 30)

    def dist_b(template, vals):
        return random.randint(5, 30)

    def dist_c(template, vals):
        total_ratio = vals['a'] + vals['b']
        base = random.randint(1, 100)
        c = base * total_ratio
        while (3 * ((vals['b'] * c) // total_ratio)) % 4 != 0:  # Ensure spent is integer
            base += 1
            c = base * total_ratio
        return c

    def dist_total_ratio(template, vals):
        return vals['a'] + vals['b']

    def dist_name_coins(template, vals):
        return (vals['b'] * vals['c']) // vals['total_ratio']

    def dist_spent(template, vals):
        return (3 * vals['name_coins']) // 4

    def dist_remaining(template, vals):
        return vals['name_coins'] - vals['spent']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_total_ratio, dist_name_coins, dist_spent, dist_remaining]
    var_names = ['name', 'a', 'b', 'c', 'total_ratio', 'name_coins', 'spent', 'remaining']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_41(generator):
    marked_q = '''{name:s} collected {a:d} starfish with 5 arms each and {b:d} seastars with 14 arms. 
How many arms do the animals she collected have in total?'''
    marked_a = '''First find the total number of starfish arms: {a:d} starfish * 5 arms/starfish = <<{a:d}*5={starfish_arms:d}>>{starfish_arms:d} arms.
Then add the number of seastar arms to find the total number of arms: {starfish_arms:d} arms + {b:d} * 14 arms = <<{starfish_arms:d}+({b:d}*14)={total_arms:d}>>{total_arms:d} arms.
#### {total_arms:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 20)

    def dist_b(template, vals):
        return random.randint(5, 20)

    def dist_starfish_arms(template, vals):
        return vals['a'] * 5

    def dist_total_arms(template, vals):
        return vals['starfish_arms'] + vals['b'] * 14

    dists = [dist_name, dist_a, dist_b, dist_starfish_arms, dist_total_arms]
    var_names = ['name', 'a', 'b', 'starfish_arms', 'total_arms']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

# def template_42(generator):
#     marked_q = '''{name:s} has {a:d} less apples than Martha, and Harry has 1/{c:d}th as many apples as {name:s}. 
# If Martha has {b:d} apples, how many apples does Harry have?'''
#     marked_a = '''{name:s} has {b:d} - {a:d} = <<{b:d}-{a:d}={name_apples:d}>>{name_apples:d} apples.
# Harry has {name_apples:d} / {c:d} = <<{name_apples:d}/{c:d}={harry_apples:d}>>{harry_apples:d} apples.
# #### {harry_apples:d}'''

#     def dist_name(template, vals):
#         return random.choice(generator.names)

#     def dist_a(template, vals):
#         return random.randint(5, 50)

#     def dist_b(template, vals):
#         return random.randint(vals['a'] + 1, 200)

#     def dist_c(template, vals):
#         name_apples = vals['b'] - vals['a']
#         possible_c = [i for i in range(2, 20) if name_apples % i == 0]
#         if not possible_c:
#             raise ValueError("No suitable c found")
#         return random.choice(possible_c)

#     def dist_name_apples(template, vals):
#         return vals['b'] - vals['a']

#     def dist_harry_apples(template, vals):
#         return vals['name_apples'] // vals['c']

#     dists = [dist_name, dist_a, dist_b, dist_c, dist_name_apples, dist_harry_apples]
#     var_names = ['name', 'a', 'b', 'c', 'name_apples', 'harry_apples']
#     return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_43(generator):
    marked_q = '''At a flea market, {name:s} sells handmade crafts for {b:d} dollars per craft. 
Today, {name:s} sells {a:d} crafts and is given an extra {c:d} dollars from an appreciative customer. 
Later on, {name:s} deposits {d:d} dollars from today's profits into her bank account. 
How many dollars is {name:s} left with after making the deposit?'''
    marked_a = '''{name:s} sells {a:d} crafts for {b:d} dollars each, for a total of {a:d} crafts * ${b:d}/craft = $<<{a:d}*{b:d}={craft_earnings:d}>>{craft_earnings:d}.
She receives an extra {c:d} dollars from a customer, increasing the total to ${craft_earnings:d} + ${c:d} = $<<{craft_earnings:d}+{c:d}={total_before_deposit:d}>>{total_before_deposit:d}.
She then deposits {d:d} dollars in the bank, leaving her with ${total_before_deposit:d} - ${d:d} = $<<{total_before_deposit:d}-{d:d}={remaining:d}>>{remaining:d}.
#### {remaining:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 20)

    def dist_b(template, vals):
        return random.randint(5, 50)

    def dist_c(template, vals):
        return random.randint(5, 50)

    def dist_craft_earnings(template, vals):
        return vals['a'] * vals['b']

    def dist_total_before_deposit(template, vals):
        return vals['craft_earnings'] + vals['c']

    def dist_d(template, vals):
        return random.randint(5, vals['total_before_deposit'] - 1)

    def dist_remaining(template, vals):
        return vals['total_before_deposit'] - vals['d']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_craft_earnings, dist_total_before_deposit, dist_d, dist_remaining]
    var_names = ['name', 'a', 'b', 'c', 'craft_earnings', 'total_before_deposit', 'd', 'remaining']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_44(generator):
    marked_q = '''{name:s} is filling an aquarium for her fish. 
She fills it halfway and goes to answer the door. 
While she's gone, her cat knocks the aquarium over and spills half the water in it. 
Then {name:s} comes back and triples the amount of water in the aquarium. 
If the aquarium is {a:d} feet long, {b:d} feet wide, and {c:d} feet high, how many cubic feet of water are in the aquarium?'''
    marked_a = '''First calculate the volume of the aquarium by multiplying its length, width and height: {a:d} ft * {b:d} ft * {c:d} ft = <<{a:d}*{b:d}*{c:d}={volume:d}>>{volume:d} cubic ft.
Then figure out what proportion of the aquarium is full after the cat knocks it over: 1/2 * 1/2 = 1/4.
Then figure out what proportion of the aquarium is full after {name:s} refills it: 3 * 1/4 = 3/4.
Now multiply the proportion of the aquarium that's full by the aquarium's volume to find out how much water is in it: {volume:d} cubic ft * 3/4 = <<{volume:d}*3/4={water_volume:d}>>{water_volume:d} cubic ft.
#### {water_volume:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 20)

    def dist_b(template, vals):
        return random.randint(5, 50)

    def dist_c(template, vals):
        return random.randint(5, 50)

    def dist_volume(template, vals):
        return vals['a'] * vals['b'] * vals['c']

    def dist_water_volume(template, vals):
        return (vals['volume'] * 3) // 4

    dists = [dist_name, dist_a, dist_b, dist_c, dist_volume, dist_water_volume]
    var_names = ['name', 'a', 'b', 'c', 'volume', 'water_volume']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_45(generator):
    marked_q = '''It is {name:s}'s turn to provide a snack for the baseball team after the game and he has decided to bring trail mix. 
The trail mix comes in packs of {a:d} individual pouches. {name:s} has {b:d} members on his baseball team, plus {c:d} coaches and {d:d} helpers. 
How many packs of trail mix does he need to buy?'''
    marked_a = '''{name:s} will need {b:d} + {c:d} + {d:d} = <<{b:d}+{c:d}+{d:d}={total_pouches:d}>>{total_pouches:d} pouches of trail mix.
If you divide the amount of trail mix pouches by the amount in each pack, you need {total_pouches:d} / {a:d} = <<{total_pouches:d}/{a:d}={packs:d}>>{packs:d} packs of trail mix.
#### {packs:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 20)

    def dist_b(template, vals):
        return random.randint(5, 50)

    def dist_c(template, vals):
        return random.randint(5, 20)

    def dist_d(template, vals):
        return random.randint(5, 20)

    def dist_total_pouches(template, vals):
        return vals['b'] + vals['c'] + vals['d']

    def dist_packs(template, vals):
        return (vals['total_pouches'] + vals['a'] - 1) // vals['a']  # Ceiling division

    dists = [dist_name, dist_a, dist_b, dist_c, dist_d, dist_total_pouches, dist_packs]
    var_names = ['name', 'a', 'b', 'c', 'd', 'total_pouches', 'packs']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_46(generator):
    marked_q = '''Four people lost a total of {a:d} kilograms of weight. 
The first person lost {b:d} kilograms. 
The second person lost {c:d} kilograms less than the first person. 
The two remaining people lost the same amount. 
How many kilograms did each of the last two people lose?'''
    marked_a = '''Second person lost {b:d} - {c:d} = <<{b:d}-{c:d}={second_person:d}>>{second_person:d} kg.
{a:d} - {b:d} - {second_person:d} = <<{a:d}-{b:d}-{second_person:d}={remaining_weight:d}>>{remaining_weight:d} kg.
{remaining_weight:d} / 2 = <<{remaining_weight:d}/2={each_last_two:d}>>{each_last_two:d} kg.
The last two people each lost {each_last_two:d} kilograms of weight.
#### {each_last_two:d}'''

    def dist_b(template, vals):
        return random.randint(25, 50)

    def dist_c(template, vals):
        return random.randint(5, vals['b'] - 1)

    def dist_a(template, vals):
        second_person = vals['b'] - vals['c']
        remaining_weight = random.randint(10, 100) * 2  # Ensure divisible by 2
        return vals['b'] + second_person + remaining_weight

    def dist_second_person(template, vals):
        return vals['b'] - vals['c']

    def dist_remaining_weight(template, vals):
        return vals['a'] - vals['b'] - vals['second_person']

    def dist_each_last_two(template, vals):
        return vals['remaining_weight'] // 2

    dists = [dist_b, dist_c, dist_a, dist_second_person, dist_remaining_weight, dist_each_last_two]
    var_names = ['b', 'c', 'a', 'second_person', 'remaining_weight', 'each_last_two']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_47(generator):
    marked_q = '''{name:s} and Andrew had breakfast at a cafe. A slice of toast costs ${a:d}, and eggs cost ${b:d} each. 
{name:s} had {c:d} slices of toast and {d:d} eggs. 
Andrew had {e:d} slices of toast and {f:d} eggs. 
How much did their breakfast cost?'''
    marked_a = '''The cost of {name:s}'s toast is {c:d} x ${a:d} = $<<{c:d}*{a:d}={name_toast:d}>>{name_toast:d}.
The cost of Andrew's toast is {e:d} x ${a:d} = $<<{e:d}*{a:d}={andrew_toast:d}>>{andrew_toast:d}.
The cost of {name:s}'s eggs is {d:d} x ${b:d} = $<<{d:d}*{b:d}={name_eggs:d}>>{name_eggs:d}.
The cost of Andrew's eggs is {f:d} x ${b:d} = $<<{f:d}*{b:d}={andrew_eggs:d}>>{andrew_eggs:d}.
Their breakfast cost ${name_toast:d} + ${andrew_toast:d} + ${name_eggs:d} + ${andrew_eggs:d} = $<<{name_toast:d}+{andrew_toast:d}+{name_eggs:d}+{andrew_eggs:d}={total_cost:d}>>{total_cost:d}.
#### {total_cost:d}'''

    def dist_name(template, vals):
        return random.choice(generator.names)

    def dist_a(template, vals):
        return random.randint(5, 20)

    def dist_b(template, vals):
        return random.randint(5, 20)

    def dist_c(template, vals):
        return random.randint(2, 10)

    def dist_d(template, vals):
        return random.randint(2, 10)

    def dist_e(template, vals):
        return random.randint(2, 10)

    def dist_f(template, vals):
        return random.randint(2, 10)

    def dist_name_toast(template, vals):
        return vals['c'] * vals['a']

    def dist_andrew_toast(template, vals):
        return vals['e'] * vals['a']

    def dist_name_eggs(template, vals):
        return vals['d'] * vals['b']

    def dist_andrew_eggs(template, vals):
        return vals['f'] * vals['b']

    def dist_total_cost(template, vals):
        return vals['name_toast'] + vals['andrew_toast'] + vals['name_eggs'] + vals['andrew_eggs']

    dists = [dist_name, dist_a, dist_b, dist_c, dist_d, dist_e, dist_f, dist_name_toast, dist_andrew_toast, dist_name_eggs, dist_andrew_eggs, dist_total_cost]
    var_names = ['name', 'a', 'b', 'c', 'd', 'e', 'f', 'name_toast', 'andrew_toast', 'name_eggs', 'andrew_eggs', 'total_cost']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

def template_48(generator):
    marked_q = '''A garden produced {a:d} potatoes, {b:d} fewer cucumbers and {c:d} times as many peppers as cucumbers.
How many vegetables did the garden produce?'''
    marked_a = '''The garden produced {a:d} potatoes - {b:d} = <<{a:d}-{b:d}={cucumbers:d}>>{cucumbers:d} cucumbers.
The garden produced {cucumbers:d} * {c:d} = <<{cucumbers:d}*{c:d}={peppers:d}>>{peppers:d} peppers.
The garden produced {a:d} potatoes + {cucumbers:d} cucumbers + {peppers:d} peppers = <<{a:d}+{cucumbers:d}+{peppers:d}={total_vegetables:d}>>{total_vegetables:d} vegetables.
#### {total_vegetables:d}'''

    def dist_a(template, vals):
        return random.randint(100, 250)

    def dist_b(template, vals):
        return random.randint(5, vals['a'] - 1)

    def dist_c(template, vals):
        return random.randint(2, 10)

    def dist_cucumbers(template, vals):
        return vals['a'] - vals['b']

    def dist_peppers(template, vals):
        return vals['cucumbers'] * vals['c']

    def dist_total_vegetables(template, vals):
        return vals['a'] + vals['cucumbers'] + vals['peppers']

    dists = [dist_a, dist_b, dist_c, dist_cucumbers, dist_peppers, dist_total_vegetables]
    var_names = ['a', 'b', 'c', 'cucumbers', 'peppers', 'total_vegetables']
    return {'marked_q': marked_q, 'marked_a': marked_a, 'dists': dists, 'var_names': var_names}

names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Avery", "Riley", "Jamie", 
         "Quinn", "Parker", "Rowan", "Skylar", "Reese", "Drew", "Emerson", "Hayden", 
         "Charlie", "Blair", "Sawyer", "Dakota", "Finley", "Arden", "Jules", "Micah", 
         "Sage", "River", "Sam", "Shay", "Elliot", "Phoenix", "Leslie", "Terry", 
         "Robin", "Cameron", "Kai", "Harper", "Payton", "Devon", "Hollis", "Blake", 
         "Remy", "Toby", "Kennedy", "Adrian", "August", "Ari", "Lennon", "Marley", 
         "Brett", "Francis", "Kendall", "Eden", "Rowe", "Monroe", "Shiloh", "Ash", 
         "Marlowe", "Ellis", "Case", "Aspen", "London", "Justice", "Jesse", "Alden", 
         "Linden", "Robin", "Tatum", "Wren", "Mackenzie", "Addison", "Luca", "Keegan", 
         "Oakley", "Sky", "Spencer", "Chandler", "Sloan", "Darcy", "Reagan", "Frankie", 
         "Rory", "Leighton", "Perry", "Gray", "Lane", "Arlo", "Ellery", "Corey", 
         "Kieran", "Bailey", "Presley", "Hunter", "Shane", "Cypress", "Quincy", 
         "Haven", "Sasha", "Scout"]
names_male = ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Jamie", "Parker", "Rowan", 
              "Reese", "Drew", "Emerson", "Hayden", "Charlie", "Sawyer", "Dakota", "Finley", 
              "Micah", "Sam", "Shay", "Elliot", "Phoenix", "Terry", "Robin", "Cameron", 
              "Kai", "Payton", "Devon", "Blake", "Toby", "Adrian", "August", "Ari", 
              "Lennon", "Brett", "Francis", "Kendall", "Rowe", "Monroe", "Ash", "Case", 
              "Justice", "Jesse", "Alden", "Linden", "Luca", "Keegan", "Spencer", "Chandler", 
              "Frankie", "Rory", "Gray", "Lane", "Arlo", "Corey", "Kieran", "Hunter", "Shane", 
              "Cypress", "Quincy"]
names_female = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Avery", "Riley", "Jamie", 
                "Quinn", "Skylar", "Reese", "Emerson", "Hayden", "Blair", "Dakota", "Arden", 
                "Jules", "Sage", "River", "Shay", "Leslie", "Terry", "Robin", "Harper", 
                "Payton", "Hollis", "Kennedy", "August", "Lennon", "Marley", "Kendall", "Eden", 
                "Rowe", "Shiloh", "Marlowe", "Ellis", "Aspen", "London", "Justice", "Jesse", 
                "Tatum", "Wren", "Mackenzie", "Addison", "Oakley", "Sky", "Sloan", "Darcy", 
                "Reagan", "Frankie", "Leighton", "Sasha", "Bailey", "Presley", "Sasha", "Scout"]
male_family = ["nephew", "cousin", "brother"]
female_family = ["niece", "cousin", "sister"]

if __name__ == "__main__":
    gsm8k = load_dataset("gsm8k", "main")
    generator = Template_Generator(names, names_male, names_female, male_family, female_family, gsm8k)

    # Create templates
    generator.create_template(template_1)
    generator.create_template(template_2)
    generator.create_template(template_3)
    generator.create_template(template_4)
    generator.create_template(template_5)
    generator.create_template(template_6)
    generator.create_template(template_7)
    generator.create_template(template_8)
    generator.create_template(template_9)
    generator.create_template(template_10)
    generator.create_template(template_11)
    generator.create_template(template_12)
    generator.create_template(template_13)
    generator.create_template(template_14)
    generator.create_template(template_15)
    generator.create_template(template_16)
    generator.create_template(template_17)
    generator.create_template(template_18)
    generator.create_template(template_19)
    generator.create_template(template_20)
    generator.create_template(template_21)
    generator.create_template(template_22)
    generator.create_template(template_23)
    generator.create_template(template_24)
    generator.create_template(template_25)
    generator.create_template(template_26)
    generator.create_template(template_27)
    generator.create_template(template_28)
    generator.create_template(template_29)
    generator.create_template(template_30)
    generator.create_template(template_31)
    generator.create_template(template_32)
    generator.create_template(template_33)
    generator.create_template(template_34)
    generator.create_template(template_35)
    generator.create_template(template_36)
    generator.create_template(template_37)
    generator.create_template(template_38)
    # generator.create_template(template_39)
    generator.create_template(template_40)
    generator.create_template(template_41)
    # generator.create_template(template_42)
    generator.create_template(template_43)
    generator.create_template(template_44)
    generator.create_template(template_45)
    generator.create_template(template_46)
    generator.create_template(template_47)
    generator.create_template(template_48)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50, help="Number of samples for each template.")
    args = parser.parse_args()

    # Generate and store questions
    directory = os.getcwd()
    output_file = f"unified_gsm8k_n={args.n}.json"
    generator.store_templates(directory, output_file, args.n)