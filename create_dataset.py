"""
SmartHire Training Dataset Generator
====================================
Script: create_dataset.py
Project: SmartHire - AI-Powered Automated Technical Interview & Evaluation Platform

Overview for Viva / Academic Presentation:
------------------------------------------
In Natural Language Processing (NLP) systems designed for educational assessment
and automated interviewing, training a text classifier requires labeled pairs of
(question, candidate_answer) -> quality_label.

This script implements a synthetic data generation pipeline that:
1. Ingests structured questions from `data/question_bank.json`.
2. Generates candidate answers across 4 distinct quality tiers (ordinal scale 0 to 3):
   - Label 0 (Poor): Short (1-5 words), irrelevant, anxious, or 'I don't know' responses.
   - Label 1 (Needs Improvement): Vague, shallow, 1-sentence answers lacking technical depth.
   - Label 2 (Good): Correct, structured, 2-3 sentences mentioning key concepts.
   - Label 3 (Excellent): Comprehensive, 3-5 sentences with architectural depth, edge cases,
                          and senior-level precision.
3. Injects controlled linguistic variation using stylistic templates and seeded randomization
   to ensure the classifier learns semantic quality rather than rote template matching.
4. Exports the resulting stratified dataset to `data/interview_dataset.csv`.
5. Computes and prints dataset balance, lexical metrics, and validation summaries.

Key Concepts for Students:
--------------------------
- Ordinal Classification: Labels [0, 1, 2, 3] have a natural ordering of quality.
- Lexical & Semantic Diversity: Varying sentence openers, vocabulary, and grammar prevents overfitting.
- Deterministic Reproducibility: Using `random.seed(42)` guarantees identical dataset generation
  across different environments and test runs.
- Class Balance: Uniform distribution (45 questions * 4 levels = 180 rows, exactly 25% per class)
  avoids majority-class prior bias during model training.
"""

import os
import json
import random
import csv
from typing import Dict, List, Any, Tuple


# ==============================================================================
# Configuration & Constants
# ==============================================================================
RANDOM_SEED: int = 42

# Resolve file paths relative to this script's directory for robust execution
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(SCRIPT_DIR, "data")
QUESTION_BANK_PATH: str = os.path.join(DATA_DIR, "question_bank.json")
OUTPUT_CSV_PATH: str = os.path.join(DATA_DIR, "interview_dataset.csv")

# ==============================================================================
# Synthetic Answer Templates & Variations
# ==============================================================================

# Label 0: Nervous, unprepared, or evasive responses (1 - 5 words)
POOR_ANSWER_TEMPLATES: List[str] = [
    "I don't know honestly.",
    "Not really sure, sorry.",
    "No idea about this.",
    "Skip this question please.",
    "Haven't used this before.",
    "Uh, something with coding?",
    "Can we move on?",
    "I forgot the concept.",
    "Never heard of it.",
    "Maybe it's for performance?",
    "Pass to next question.",
    "Not familiar with this.",
    "I think it's functions?",
    "No clue right now.",
    "I don't recall this.",
    "Just basic computer logic.",
    "I haven't studied this.",
    "Could we skip please?",
    "Blanking out on this.",
    "Not sure at all."
]

# Stylistic prefixes to add natural conversational variety to Good & Excellent answers
GOOD_OPENERS: List[str] = [
    "In general, ",
    "From my understanding, ",
    "Basically, ",
    "In practical terms, ",
    "Typically, ",
    "In software development, ",
    "To explain this clearly, "
]

EXCELLENT_OPENERS: List[str] = [
    "From an engineering perspective, ",
    "In production architecture, ",
    "Fundamentally, ",
    "At a systems level, ",
    "In professional software engineering, ",
    "To thoroughly address this, "
]

# Question-specific answer templates mapping question index / theme to tailored responses
# Format: role -> difficulty -> list of 5 question responses for labels [1, 2, 3]
TAILORED_RESPONSES: Dict[str, Dict[str, List[Dict[int, str]]]] = {
    "python_developer": {
        "easy": [
            # Q1: Mutable vs Immutable
            {
                1: "Mutable types like lists can be changed after creation, while immutable types like strings or tuples cannot be changed.",
                2: "Mutable objects such as lists and dictionaries can have their contents modified in place without changing memory address. In contrast, immutable objects like tuples and integers cannot be modified once instantiated, meaning any change creates a new object. Knowing this difference is important to avoid unexpected side effects in function arguments.",
                3: "In Python, mutable objects like lists, sets, and dictionaries allow in-place modification of their values while maintaining a constant memory address (id). Conversely, immutable objects such as strings, integers, and tuples cannot be altered after creation, forcing Python to allocate a brand-new object in memory whenever modifications are attempted. Passing mutable objects as default arguments to functions is a notorious pitfall that causes unintended shared state across calls. In production systems, immutable data structures are often preferred in multi-threaded workflows to guarantee thread-safe data integrity."
            },
            # Q2: List Comprehensions
            {
                1: "List comprehensions are a shorter way to write for loops when you want to filter or make a new list.",
                2: "List comprehensions provide a concise syntactic shortcut for creating lists by iterating over an iterable with optional if-conditions. They are generally preferred over standard for loops because they reduce boilerplate and run faster at the bytecode level in CPython. However, traditional loops are better when the logic has complex nested conditions.",
                3: "List comprehensions provide an elegant, expressive syntactic construct for mapping and filtering elements from an iterable into a new list. Under the hood in CPython, list comprehensions execute optimized specialized bytecode (LIST_APPEND) that outperforms explicit for-loop appends. However, when comprehension expressions become deeply nested or incorporate complex branching, standard loops are strongly advised to preserve codebase readability. Furthermore, for very large datasets, generator expressions should be selected instead to preserve memory through lazy evaluation."
            },
            # Q3: == vs is
            {
                1: "The == operator checks if the values are equal, while the is operator checks if they are the exact same object in memory.",
                2: "The == operator compares value equality by calling an object's __eq__() method to verify if contents match. Meanwhile, the is operator tests identity by checking whether two references share the exact same memory address using their id(). For instance, two distinct lists with identical elements return True with == but False with is.",
                3: "The == operator tests value equivalence by invoking the underlying __eq__() magic method, checking whether two variables possess identical data regardless of memory location. Conversely, the is operator assesses object identity, validating whether both variables point to the identical memory address via their id() pointer. For example, two distinct list literals with identical contents evaluate to True with == but evaluate to False with is. Best practice strictly dictates using the is operator for singleton comparisons, most notably when checking if an object is None."
            },
            # Q4: __init__ vs __new__
            {
                1: "__init__ initializes the object attributes after __new__ creates the class instance.",
                2: "The __init__ method is the initializer that sets up instance attributes and state on self, returning None. In contrast, __new__ is the actual constructor method that allocates memory and returns a new instance of the class. Most developers only override __init__, reserving __new__ for immutable subclasses or metaclass logic.",
                3: "In Python's object lifecycle, __new__ is the true static constructor responsible for allocating memory and returning a freshly created instance of the class. Once __new__ produces the instance, Python automatically passes that instance as self to the __init__ initializer method to bind attributes and set up initial state. While __init__ returns None, __new__ must explicitly return an object instance to trigger __init__. Custom overrides of __new__ are primarily required when subclassing immutable built-in types like int or str, or when implementing singleton patterns."
            },
            # Q5: Error handling (try, except, else, finally)
            {
                1: "try runs the code, except catches errors, else runs if there were no errors, and finally always runs.",
                2: "Structured exception handling uses try blocks to guard dangerous code and except blocks to capture specific exceptions. The else block executes only if the try block succeeded without error, which is useful for actions that shouldn't be guarded. The finally block executes unconditionally regardless of errors, making it ideal for cleanup tasks like closing files.",
                3: "Python's structured exception handling executes risky operations within a try block and intercepts specified exceptions through targeted except clauses. The optional else block executes strictly when the try block completes without encountering any exceptions, preventing accidental catching of unrelated errors. Crucially, the finally block is guaranteed to execute under all conditions, ensuring reliable resource cleanup such as closing database connections or socket descriptors. Industry best practice mandates catching precise exception classes rather than using bare except clauses."
            }
        ],
        "medium": [
            # Q6: Generators and yield
            {
                1: "Generators use yield instead of return so you can iterate over items one at a time without saving everything in memory.",
                2: "Generators are functions that implement the iterator protocol using the yield keyword to produce values lazily on demand. When yield is hit, execution pauses and the local state is preserved until the next iteration. This keeps memory usage at O(1) because elements are generated one by one instead of loading full collections into RAM.",
                3: "Generators are specialized iterator functions that leverage the yield keyword to pause execution and produce values lazily upon request. Unlike standard functions that return an entire materialized sequence, generators yield values on the fly, maintaining an O(1) memory footprint regardless of dataset magnitude. When yield is executed, the function's stack frame and local variables are frozen in memory until the consumer calls next(). This makes generators essential for streaming large log files, database cursor chunking, and handling infinite pipelines without triggering Out-Of-Memory exceptions."
            },
            # Q7: Decorators and closures
            {
                1: "Decorators are functions that wrap other functions using the @ symbol to add extra behavior like logging or auth.",
                2: "A decorator is a higher-order function that takes another function as input, wraps it with additional logic using a closure, and returns the modified function. They rely on Python functions being first-class citizens and use @decorator syntax as syntactic sugar. Common use cases include timing execution, logging, and user authentication.",
                3: "In Python, decorators are higher-order functions that take a callable as an argument and return a modified callable enriched with supplementary behavior via lexical closures. The @decorator syntax is syntactic sugar for func = decorator(func), allowing engineers to adhere to the Open/Closed Principle by extending functionality without mutating existing code. Practical enterprise use cases include request authentication, distributed tracing, memoization caching, and input validation. To avoid obscuring the original function's name and docstring, custom decorators must always employ the @functools.wraps decorator internally."
            },
            # Q8: GIL (Global Interpreter Lock)
            {
                1: "The GIL is a lock in Python that prevents multiple threads from running Python code at the same time on multiple cores.",
                2: "The Global Interpreter Lock (GIL) is a mutex in CPython that ensures only one native OS thread executes Python bytecode at any moment. While it simplifies memory management and reference counting, it prevents CPU-bound multi-threaded scripts from utilizing multi-core processors. However, I/O-bound operations release the GIL during network or disk waiting, so threading still works well for I/O.",
                3: "The Global Interpreter Lock (GIL) is a mutual exclusion lock implemented in CPython to prevent race conditions during reference-counting memory management operations. Because the GIL permits only one native thread to execute Python bytecode at any given moment, CPU-bound workloads cannot achieve true hardware parallelism across multiple processor cores. In contrast, I/O-bound tasks release the GIL while waiting on network sockets or disk operations, enabling effective concurrency through threading or asyncio. To achieve true multi-core parallel processing for computational workloads, Python engineers rely on the multiprocessing module."
            },
            # Q9: Garbage collection & reference counting
            {
                1: "Python tracks how many references point to an object, and a cyclic garbage collector cleans up circular references.",
                2: "Python's primary memory management is reference counting, where an object is deallocated immediately when its reference count drops to zero. To resolve circular references where objects reference each other, Python includes a generational cyclic garbage collector. This collector runs across three generations to detect and clean unreferenced reference cycles.",
                3: "Python manages runtime memory primarily through reference counting: every object header tracks active incoming references, deallocating the object instantly when its count drops to zero. To prevent memory leaks caused by circular references where objects reference each other, CPython utilizes a generational cyclic garbage collector. This subsystem categorizes objects into three generations based on survival tenure and executes cycle-detection heuristics periodically. Engineers can fine-tune collection thresholds, force collections, or disable the cyclic collector in latency-critical batch pipelines using the built-in gc module."
            },
            # Q10: *args and **kwargs
            {
                1: "*args accepts variable positional arguments as a tuple, and **kwargs accepts variable keyword arguments as a dictionary.",
                2: "In Python functions, *args allows capturing any number of positional arguments into a tuple, while **kwargs captures keyword arguments into a dictionary. They are widely used in decorator wrappers, API forwarding, and subclass constructors calling super(). At the call site, * and ** unpack iterables and mappings into separate arguments.",
                3: "In Python function signatures, *args allows a callable to accept an arbitrary quantity of positional parameters packed into a tuple, while **kwargs accepts arbitrary keyword arguments packed into a dictionary. They are indispensable for constructing wrapper functions, maintaining backward-compatible APIs, and dispatching arguments to superclass initializers via super(). At call sites, the unary * and ** operators perform sequence and dictionary unpacking into function parameters. In parameter definitions, standard positional arguments must precede *args, followed by keyword-only parameters and finally **kwargs."
            }
        ],
        "hard": [
            # Q11: Metaclasses
            {
                1: "Metaclasses are classes that define how classes themselves are created, with type being the default metaclass.",
                2: "In Python, classes are themselves instances of metaclasses, where type is the standard default metaclass. By subclassing type, you can intercept class creation by overriding __new__ and __init__ before instances are created. They are useful for enforcing API constraints or auto-registering plugins, though modern Python often uses __init_subclass__ instead.",
                3: "In Python's unified object model, classes are themselves first-class objects instantiated by metaclasses, with the built-in type acting as the universal default metaclass. By constructing a custom metaclass subclassing type, architects can intercept and mutate the class construction pipeline by overriding __new__ and __init__ prior to class object instantiation. This provides deep metaprogramming capabilities such as validating class attributes, enforcing design patterns, and automatically registering plugins. While modern Python often favors simpler alternatives like __init_subclass__ or class decorators, metaclasses underpin foundational frameworks like ORMs and schema engines."
            },
            # Q12: Asynchronous programming with asyncio
            {
                1: "Asyncio uses an event loop and coroutines with async and await to run non-blocking network requests efficiently.",
                2: "Python's asyncio provides cooperative multitasking using a single-threaded event loop that schedules and runs coroutines. Coroutines are declared with async def and use await to yield control back to the event loop while waiting for non-blocking I/O. This allows a single process to handle thousands of concurrent network connections without thread overhead.",
                3: "Python's asyncio framework implements cooperative multitasking via a single-threaded event loop that schedules and executes non-blocking coroutines. Coroutines declared with async def pause their execution at await expressions, yielding control back to the event loop so other concurrent tasks can run while awaiting I/O completion. This architectural pattern avoids the severe memory footprint and context-switching overhead of OS-level threads, enabling tens of thousands of concurrent connections. However, CPU-intensive or blocking synchronous operations will stall the entire event loop, necessitating offloading via loop.run_in_executor()."
            },
            # Q13: Descriptors
            {
                1: "Descriptors are objects that define __get__, __set__, or __delete__ to customize how class attributes are accessed.",
                2: "Descriptors are Python objects that implement the descriptor protocol methods: __get__, __set__, and __delete__. When an attribute on an instance is accessed, Python checks if it is a descriptor and invokes the corresponding method instead of standard dict lookup. Built-in features like @property, @classmethod, and ORM fields are all built on descriptors.",
                3: "Descriptors are objects that customize attribute lookup, assignment, and deletion behavior by implementing the descriptor protocol: __get__, __set__, and __delete__. Data descriptors implement __set__ or __delete__ and take precedence over instance dictionary lookup, whereas non-data descriptors implement only __get__ (such as standard methods). When an attribute is accessed on an instance, Python's internal lookup mechanism detects the descriptor and invokes its protocol method. Core Python features such as @property, @classmethod, @staticmethod, and modern ORM column abstractions are built directly on descriptors."
            },
            # Q14: Method Resolution Order (MRO)
            {
                1: "MRO is the order Python searches parent classes during multiple inheritance, calculated using the C3 linearization algorithm.",
                2: "Method Resolution Order (MRO) defines the exact sequence Python follows when looking up attributes and methods in inheritance hierarchies. Python uses the C3 Linearization algorithm to resolve multiple inheritance and avoid the diamond problem. It ensures child classes precede parents and maintains the order parents were declared when using super().",
                3: "Method Resolution Order (MRO) is the deterministic sequence Python utilizes to resolve attribute and method lookups in inheritance graphs, viewable via the __mro__ attribute. Python computes this order using the C3 Linearization algorithm, which effectively resolves the classic diamond inheritance dilemma. C3 strictly guarantees two invariants: local precedence order matching the declaration sequence and monotonicity across subclass-superclass hierarchies. Mastering MRO is critical when engineering cooperative multiple inheritance architectures that utilize super() to dispatch method calls without duplicate execution."
            },
            # Q15: __slots__
            {
                1: "Slots replace the instance __dict__ with a fixed array of attributes to reduce memory usage when creating many objects.",
                2: "By default, Python objects store their attributes in an internal __dict__, which has significant memory overhead. Defining __slots__ with a tuple of attribute names replaces this dict with a fixed-size array of descriptors, saving substantial memory when instantiating millions of objects. However, you cannot add dynamic attributes that were not defined in slots.",
                3: "By default, Python instances store instance attributes inside an internal dictionary (__dict__), which introduces significant memory overhead when instantiating millions of small objects. Declaring __slots__ with an iterable of attribute names replaces the dynamic __dict__ with a fixed-size C-level array of descriptor pointers, dramatically reducing per-instance memory consumption and accelerating attribute lookup speeds. However, __slots__ prevents dynamic assignment of undeclared attributes at runtime unless '__dict__' is explicitly included in the slots declaration. Additionally, inheritance requires caution because subclasses do not inherit slots automatically from parent classes."
            }
        ]
    },
    "data_analyst": {
        "easy": [
            # Q1: WHERE vs HAVING
            {
                1: "WHERE filters rows before grouping data, and HAVING filters rows after the GROUP BY aggregation.",
                2: "In SQL, the WHERE clause filters individual records before any grouping or aggregation takes place. The HAVING clause is applied after the GROUP BY clause to filter aggregated metric results, such as COUNT() or SUM(). Because WHERE cannot evaluate aggregate functions, HAVING must be used when filtering on group summaries.",
                3: "In SQL querying, the WHERE clause filters individual records before grouping and aggregation occur, evaluating boolean expressions against raw table rows. Conversely, the HAVING clause is executed after the GROUP BY stage, specifically designed to filter aggregated groups based on summary functions such as COUNT(), SUM(), or AVG(). Because the WHERE clause is processed earlier in the SQL execution pipeline, it cannot evaluate aggregate calculations directly. Optimizing query performance requires placing row-level filters in WHERE to prune dataset volume before grouping, leaving only necessary group aggregations for HAVING."
            },
            # Q2: SQL Joins
            {
                1: "Inner join returns matching rows from both tables, while left join returns all rows from the left table and matches from the right.",
                2: "An INNER JOIN returns only records where there is a match between both joined tables. A LEFT JOIN preserves all records from the left table and includes matching right table columns, filling in NULL when there is no match. A FULL OUTER JOIN keeps all records from both tables regardless of matches, placing NULLs where keys don't align.",
                3: "Relational SQL joins combine records from two or more tables based on defined relational keys. An INNER JOIN filters results strictly to rows that satisfy the join condition in both tables, discarding unmatched records. A LEFT JOIN retains all records from the primary left table alongside matching rows from the right table, populating NULLs for right-table attributes whenever matches are absent. A FULL OUTER JOIN retains all records across both datasets, generating NULLs on either side when join keys do not correspond."
            },
            # Q3: Mean, median, mode
            {
                1: "Mean is the average, median is the middle number, and mode is the most common number, with mean being sensitive to outliers.",
                2: "Mean is calculated by dividing the sum of values by the count, making it vulnerable to extreme outliers. The median is the 50th percentile value in sorted data, providing a robust measure of central tendency for skewed distributions. The mode is the value that occurs most frequently, which is particularly useful for categorical data.",
                3: "Mean, median, and mode represent the three fundamental measures of central tendency in descriptive statistics. The arithmetic mean represents the center of mass of a distribution but suffers from high sensitivity to extreme outliers and heavy skewness. The median identifies the 50th percentile middle value of an ordered dataset, serving as a robust statistical metric resistant to distortion in skewed domains such as income or real estate prices. The mode reflects the most frequent discrete value and is especially valuable for analyzing categorical distributions."
            },
            # Q4: .loc vs .iloc
            {
                1: "loc selects rows and columns using their labels, while iloc selects them using integer positions.",
                2: "In pandas, .loc is label-based indexing where you reference rows and columns by their index names, and slice endpoints are inclusive. In contrast, .iloc is integer-based indexing that references data by 0-based positional indices with exclusive stop slices. Using the right one prevents accidental indexing errors during data transformation.",
                3: "In pandas DataFrame manipulation, .loc is a label-based indexer that selects rows and columns via explicit index labels or boolean condition masks, with slice boundaries being fully inclusive of both start and stop labels. In contrast, .iloc is a positional integer-based indexer that accesses elements strictly by their 0-indexed integer coordinates, adhering to standard Python slicing where the end index is exclusive. When working with non-integer indices like timestamps or UUIDs, .loc accesses items by key, whereas .iloc continues positional integer indexing. Selecting the explicit indexer avoids subtle indexing bugs during automated ETL pipelines."
            },
            # Q5: Missing values in pandas
            {
                1: "You can handle missing values in pandas by dropping them with dropna() or filling them with fillna() using mean or median.",
                2: "Handling missing values in pandas involves either removing nulls with dropna() or imputing them using fillna(). Imputation can use statistical metrics like mean or median for numerical features, or mode for categorical features. For sequential time series data, forward-fill (ffill) or interpolation is commonly used to preserve trends.",
                3: "Managing missing values in pandas begins with diagnosing the underlying missingness mechanism: Missing Completely at Random (MCAR), Missing at Random (MAR), or Missing Not at Random (MNAR). Direct deletion using .dropna() removes incomplete records, but risks discarding valuable information when missingness is widespread. Imputation using .fillna() substitutes missing values with summary statistics such as the median for skewed numerical data, or the mode for categorical data. For longitudinal or time-series data, forward filling (ffill) or spline interpolation preserves sequential continuity without introducing synthetic variance."
            }
        ],
        "medium": [
            # Q6: SQL Window functions
            {
                1: "Window functions calculate across rows using OVER() and PARTITION BY, with ROW_NUMBER giving unique numbers and RANK skipping ties.",
                2: "SQL window functions compute values over a defined set of rows without collapsing them into a single row like GROUP BY does. ROW_NUMBER() assigns a unique sequential integer to every row. RANK() assigns the same rank to ties but skips subsequent numbers, while DENSE_RANK() assigns the same rank to ties without skipping numbers.",
                3: "SQL window functions perform analytical calculations across a partitioned subset of rows without collapsing the underlying dataset into grouped aggregates. The OVER() clause defines the analytical frame, commonly partitioned via PARTITION BY and ordered via ORDER BY. ROW_NUMBER() assigns an unbroken unique sequence of integers to every row regardless of duplicate values. RANK() gives tied rows identical ranks but skips subsequent rank numbers by the tie count, whereas DENSE_RANK() assigns identical ranks to ties without skipping subsequent sequential values."
            },
            # Q7: Normalization vs Denormalization
            {
                1: "Normalization splits tables to avoid duplicate data in OLTP, while denormalization combines tables for faster reads in OLAP.",
                2: "Data normalization organizes database schemas into normal forms (1NF, 2NF, 3NF) to minimize redundancy and prevent update anomalies in transactional OLTP systems. Denormalization deliberately introduces redundancy by combining tables to avoid expensive joins. It is preferred in data warehousing and reporting (OLAP) to maximize query read speed.",
                3: "Database normalization organizes relational schemas into progressive normal forms (1NF through 3NF/BCNF) to eliminate data redundancy, prevent modification anomalies, and enforce referential integrity. It represents the standard paradigm for Online Transaction Processing (OLTP) environments where frequent write, update, and delete transactions demand transactional isolation. In contrast, denormalization strategically reintroduces redundant data or pre-aggregates tables to minimize costly multi-table joins. Denormalization is the architecture of choice in Online Analytical Processing (OLAP) data warehouses and reporting marts where read throughput supersedes write efficiency."
            },
            # Q8: Correlation vs Causation
            {
                1: "Correlation means two variables are related, but causation means one actually caused the other, which you test with A/B tests.",
                2: "Correlation measures the strength and direction of a linear relationship between two variables, but it does not imply that one causes the other. Spurious correlation can occur due to confounding variables or coincidence. To establish true causality, analysts run Randomized Controlled Trials (A/B testing) to isolate the treatment effect.",
                3: "Correlation measures the statistical co-movement between two continuous variables, whereas causation confirms that manipulation of an independent variable directly produces changes in a dependent variable. High correlation frequently arises from confounding variables, reverse causality, or spurious association without indicating causal influence. To establish definitive causality, analysts rely on Randomized Controlled Trials (A/B tests), where randomized assignment balances confounding factors across treatment and control cohorts. In observational studies where experimentation is impossible, quasi-experimental methods like difference-in-differences or propensity score matching are used."
            },
            # Q9: Outlier detection & handling
            {
                1: "Outliers can be found using the IQR method or Z-scores and handled by removing them or capping them with winsorization.",
                2: "Outliers can be detected using statistical methods like Z-scores (values beyond plus or minus 3 standard deviations) or the Interquartile Range (IQR) rule for skewed data. In the IQR method, values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged. Analysts handle them by verifying measurement errors, capping them with winsorization, or applying log transformations.",
                3: "Outlier detection in numerical distributions typically relies on either parametric Z-scores for Gaussian data or non-parametric Interquartile Range (IQR) boundaries for skewed distributions. Under the IQR rule, data points falling outside Q1 - 1.5*IQR or Q3 + 1.5*IQR are classified as potential anomalies. Once identified via boxplots and statistical screening, treatment depends on domain context: true measurement errors are pruned, whereas genuine extreme events are retained or normalized. Applying winsorization or monotonic log transformations stabilizes high-leverage outliers without losing sample size."
            },
            # Q10: groupby and agg in pandas
            {
                1: "groupby groups data by categories and agg lets you run multiple summary functions like sum and mean on different columns.",
                2: "The pandas groupby() method follows the split-apply-combine workflow to aggregate data across categorical groups. When paired with .agg(), you can apply multiple aggregation functions across different columns using a dictionary mapping, such as {'sales': ['sum', 'mean'], 'orders': 'count'}. The output creates a MultiIndex that can be flattened for reporting.",
                3: "The pandas groupby() method operationalizes the classic split-apply-combine paradigm, partitioning data into subsets by key columns, executing vector functions on each partition, and combining results into an output DataFrame. Pairing groupby() with the .agg() method allows analysts to run distinct aggregation functions across multiple features via dictionary mapping, such as {'revenue': ['sum', 'mean'], 'conversions': 'max'}. This aggregation creates hierarchical MultiIndex column headers that can be flattened using list comprehensions for production pipelines."
            }
        ],
        "hard": [
            # Q11: Type I & Type II errors, p-values
            {
                1: "Type I error is a false positive, Type II error is a false negative, and p-value shows if the result is statistically significant.",
                2: "A Type I error (alpha) occurs when you reject a true null hypothesis, while a Type II error (beta) occurs when you fail to reject a false null hypothesis. The p-value measures the probability of obtaining test results at least as extreme as observed, assuming the null hypothesis is true. Statistical power (1 - beta) is the likelihood of correctly detecting a true effect when one exists.",
                3: "In statistical hypothesis testing, a Type I error (alpha) represents a false positive where a true null hypothesis is incorrectly rejected, whereas a Type II error (beta) represents a false negative where an experiment fails to detect a genuine effect. The p-value quantifies the probability of observing test statistics at least as extreme as empirical findings under the assumption that the null hypothesis is true. Statistical power, formalized as 1 - beta, quantifies the probability of correctly rejecting a false null hypothesis when a true effect exists. Conducting statistical power calculations prior to A/B test launches ensures adequate sample size to detect minimum detectable effects while controlling false discovery rates."
            },
            # Q12: Complex SQL, CTEs & Indexing
            {
                1: "CTEs create readable modular subqueries using WITH, and indexes like B-trees speed up filtering and joins in query plans.",
                2: "Common Table Expressions (CTEs) define temporary named result sets using WITH, improving the modularity and readability of complex queries. Recursive CTEs allow traversal of hierarchical tree data like organizational structures. Query optimization involves running EXPLAIN to inspect execution plans and creating composite B-tree indexes matching filter and join columns.",
                3: "Common Table Expressions (CTEs) modularize complex SQL logic into maintainable, self-contained subqueries defined with WITH, and recursive CTEs enable hierarchical traversal such as organizational charts or graph trees via an anchor member and recursive union. To optimize execution, analysts inspect database query plans using EXPLAIN ANALYZE to identify bottleneck operations like full sequential table scans and excessive hash joins. Creating appropriate composite B-tree indexes matching query filter, join, and order keys significantly accelerates index-seek operations. Furthermore, understanding whether the SQL engine treats CTEs as inlined subqueries or materialized temp tables prevents unexpected memory spillage and unindexed temporary scans."
            },
            # Q13: Simpson's Paradox
            {
                1: "Simpson's Paradox is when a trend seen in different groups reverses when all the data is combined together due to a lurking variable.",
                2: "Simpson's Paradox occurs when a statistical relationship observed within distinct subgroups disappears or reverses when the groups are aggregated. This happens when an uncontrolled confounding variable influences group distributions unevenly. Analysts prevent misleading conclusions by segmenting data and calculating weighted stratified averages.",
                3: "Simpson's Paradox describes an empirical phenomenon where a statistical correlation or trend evident across separate sub-cohorts reverses or vanishes when cohorts are pooled into an aggregate dataset. This statistical distortion is typically caused by an unobserved confounding variable and disproportionate sample allocations across sub-groups. Analysts can detect Simpson's Paradox by disaggregating high-level metrics across key operational segments rather than relying on top-line averages. Mitigating this risk requires stratified analysis, weighted group averages, or multivariate regression models to isolate the authentic independent effect."
            },
            # Q14: Data leakage in ML pipelines
            {
                1: "Data leakage happens when information from the test set or future data leaks into the training data, causing unrealistically high accuracy.",
                2: "Data leakage occurs when target or future information contaminates the training set, causing inflated test metrics that fail in production. To avoid leakage, all data scaling, imputation, and encoding must be fitted strictly on training folds and only applied to validation folds. Scikit-learn Pipelines help prevent leakage during cross-validation.",
                3: "Data leakage occurs when predictive information from outside the training partition—such as the target variable or lookahead future data—contaminates model training, resulting in over-optimistic validation metrics that fail in production. To guarantee data hygiene, all preprocessing transformations including imputation, feature scaling, and categorical encoding must be fitted solely on training splits and merely transformed on test splits. Employing Scikit-Learn Pipeline architectures inside cross-validation loops guarantees that transformations are refitted independently within each cross-validation fold. For temporal datasets, strict time-based rolling window splits must be utilized rather than randomized k-fold splits to eliminate lookahead bias."
            },
            # Q15: L1 vs L2 Regularization
            {
                1: "L1 Lasso regularization shrinks unneeded feature weights to zero for feature selection, while L2 Ridge shrinks weights smoothly without zeroing them.",
                2: "L1 regularization (Lasso) adds the absolute values of coefficients to the loss function, which forces less important coefficients to become exactly zero for feature selection. L2 regularization (Ridge) adds the squared values of coefficients, shrinking weights toward zero without setting them to zero. Elastic Net combines both penalties to handle correlated features.",
                3: "Regularization penalizes large regression coefficients to mitigate model overfitting and reduce variance. L1 regularization (Lasso) penalizes the sum of the absolute values of the coefficients, geometrically producing diamond-shaped constraint boundaries that drive non-informative features to zero, functioning as automated feature selection. L2 regularization (Ridge) penalizes the sum of squared weights, yielding circular constraint surfaces that shrink coefficients asymptotically toward zero while retaining all features in the presence of multicollinearity. In machine learning workflows, Elastic Net integrates both L1 and L2 penalties to achieve balance in high-dimensional feature spaces."
            }
        ]
    },
    "web_developer": {
        "easy": [
            # Q1: DOM & JavaScript
            {
                1: "The DOM is the browser tree structure of HTML elements that JavaScript can update dynamically using methods like querySelector.",
                2: "The Document Object Model (DOM) is an object-oriented tree representation of an HTML document in browser memory. JavaScript interacts with the DOM using APIs like querySelector() and addEventListener() to manipulate elements, styles, and handle user interactions. This allows creating dynamic, responsive web pages without full page reloads.",
                3: "The Document Object Model (DOM) represents an HTML document in browser memory as an object-oriented hierarchical tree of element and text nodes. JavaScript interfaces with this tree via built-in APIs like querySelector(), createElement(), and classList to dynamically mutate attributes, structure, and styles. Furthermore, event listeners attached via addEventListener() capture user interactions like clicks, inputs, and scrolls to trigger reactive UI updates. Updating the DOM programmatically allows developers to build rich Single Page Applications without triggering full browser page reloads."
            },
            # Q2: CSS Box Model
            {
                1: "The CSS box model includes content, padding, border, and margin, and border-box makes the width include padding and borders.",
                2: "The CSS Box Model states that every HTML element is made of four areas: content, padding, border, and margin. Under the default box-sizing: content-box, padding and borders increase the total rendered width of the element. Setting box-sizing: border-box ensures declared width includes content, padding, and border, making responsive sizing much easier.",
                3: "The CSS Box Model is the fundamental geometric layout paradigm in web browsers, defining that every rendered HTML element is composed of four concentric rectangles: content, padding, border, and margin. Under the default box-sizing: content-box, an element's declared width applies strictly to the content rectangle, causing added padding and borders to expand its actual rendered footprint. Applying box-sizing: border-box recalibrates the layout model so that declared dimensions encompass content, padding, and borders simultaneously. Adopting border-box globally is an industry best practice that prevents unwanted layout overflow and streamlines responsive flexbox and grid calculations."
            },
            # Q3: Semantic HTML
            {
                1: "Semantic HTML tags like header, nav, and article give meaning to content, improving accessibility and SEO rankings.",
                2: "Semantic HTML uses meaningful tags like <header>, <main>, <nav>, and <article> instead of generic <div> tags to describe content structure. This helps screen readers and assistive technologies navigate the page for better accessibility. It also enables search engine web crawlers to understand page hierarchy for improved SEO.",
                3: "Semantic HTML refers to using markup elements that convey clear structural meaning to both web browsers and developers, such as <header>, <nav>, <main>, <article>, and <footer>, rather than generic <div> wrappers. Semantic tags are essential for web accessibility because screen readers rely on ARIA roles and landmark tags to provide accessible navigation for visually impaired users. Furthermore, search engine crawlers parse semantic structures to index core content hierarchies, directly improving Search Engine Optimization (SEO) rankings. Maintaining clean semantic markup also enhances codebase readability and maintainability across cross-functional engineering teams."
            },
            # Q4: let, const, var
            {
                1: "var is function-scoped and hoisted, while let and const are block-scoped, and const cannot be reassigned.",
                2: "In modern JavaScript, var is function-scoped and hoisted with undefined, which often causes variable leaks. In contrast, let and const are block-scoped and live in the Temporal Dead Zone until initialized. While let allows reassignment, const prevents reassigning the variable identifier, though object properties can still be mutated.",
                3: "In modern JavaScript (ES6+), legacy var declarations are function-scoped and hoisted to the top of their execution context with an initial value of undefined, frequently creating unintended scoping bugs. In contrast, let and const provide lexical block scoping within enclosing curly braces and reside within the Temporal Dead Zone (TDZ), throwing a ReferenceError if accessed prior to declaration. While let allows identifier reassignment, const enforces constant assignment to its memory reference. However, declaring objects or arrays with const does not freeze their internal values; immutability requires applying Object.freeze()."
            },
            # Q5: HTTP Request Methods
            {
                1: "GET retrieves data, POST creates new resources, PUT replaces resources, and DELETE removes them.",
                2: "HTTP request methods define actions on server resources according to REST conventions. GET is safe and idempotent for fetching data without modifying state. POST creates new resources and is non-idempotent, while PUT replaces an entire resource idempotently, PATCH applies partial changes, and DELETE removes resources.",
                3: "HTTP request methods govern client-server resource interactions under RESTful web architecture standards. GET is a safe and idempotent verb dedicated strictly to querying resources without mutating server state, whereas POST is a non-idempotent method used to submit payloads and create new entities. PUT replaces an existing resource in its entirety in an idempotent fashion, while PATCH applies targeted partial mutations to existing records. Finally, DELETE removes designated resources idempotently, ensuring predictable interaction contracts across distributed microservices and web clients."
            }
        ],
        "medium": [
            # Q6: JavaScript Event Loop
            {
                1: "The event loop manages the call stack, microtask queue for promises, and macrotask queue for setTimeout callbacks.",
                2: "JavaScript is single-threaded and uses an event loop to execute asynchronous tasks without blocking. When the synchronous call stack is empty, the event loop drains all tasks in the microtask queue (such as Promise callbacks) first. After microtasks finish, it processes one task from the macrotask queue (like setTimeout or DOM events).",
                3: "JavaScript executes on a single-threaded runtime utilizing a synchronous call stack and an asynchronous event loop to handle non-blocking concurrency. When synchronous execution completes and the call stack clears, the event loop prioritizes draining all jobs in the microtask queue, which hosts Promise resolutions (.then(), await) and MutationObservers. Only after the microtask queue is completely exhausted does the event loop dequeue a single event from the macrotask queue, which handles setTimeout, setInterval, and I/O callbacks. This execution loop guarantees non-blocking performance while maintaining deterministic ordering across asynchronous operations."
            },
            # Q7: CSR vs SSR
            {
                1: "CSR renders the page in the browser using JavaScript, while SSR generates HTML on the server for faster first loads and better SEO.",
                2: "Client-Side Rendering (CSR) downloads a minimal HTML shell and JavaScript bundle that builds the DOM in the browser, providing rich app interactions but slower initial load and weaker SEO. Server-Side Rendering (SSR) generates full HTML on the server for each request, offering superior SEO and fast First Contentful Paint. However, SSR requires client-side hydration and increases server computing overhead.",
                3: "Client-Side Rendering (CSR) delivers a bare HTML shell and executes a client JavaScript bundle to assemble the DOM dynamically in the browser, enabling smooth Single Page Application transitions but suffering from slower First Contentful Paint (FCP) and impaired SEO crawlability. Conversely, Server-Side Rendering (SSR) compiles full HTML on the server for each inbound request, delivering fully formed markup that yields instant content visibility and superior search engine indexing. However, SSR demands greater server compute infrastructure and introduces a hydration phase where JavaScript event listeners bind to static HTML before becoming fully interactive. Modern web engineering frequently leverages hybrid frameworks supporting Static Site Generation (SSG) and Incremental Static Regeneration (ISR) to optimize these performance trade-offs."
            },
            # Q8: CORS
            {
                1: "CORS is a browser security rule that blocks requests to different origins unless the backend sends Access-Control-Allow-Origin headers.",
                2: "Cross-Origin Resource Sharing (CORS) is a browser security mechanism based on the Same-Origin Policy that restricts requests to different domains, ports, or protocols. For non-simple requests, browsers send a preflight OPTIONS request to verify server permissions. Backend servers resolve CORS by setting Access-Control-Allow-Origin and related headers.",
                3: "Cross-Origin Resource Sharing (CORS) is a browser-enforced security mechanism rooted in the Same-Origin Policy that prevents web applications on one origin from accessing resources on a different domain, port, or protocol. For HTTP requests utilizing non-simple methods or custom headers, the browser automatically dispatches an initial preflight OPTIONS request to verify permissions before transmitting the actual request. Resolving CORS errors requires configuring the target backend server to return appropriate HTTP headers, including Access-Control-Allow-Origin, Access-Control-Allow-Methods, and Access-Control-Allow-Headers. Client-side workarounds cannot bypass CORS; proper resolution mandates backend header configuration or routing through an intermediary reverse proxy during local development."
            },
            # Q9: Flexbox vs CSS Grid
            {
                1: "Flexbox is one-dimensional for single rows or columns, while CSS Grid is two-dimensional for managing rows and columns together.",
                2: "CSS Flexbox is a one-dimensional layout system ideal for aligning items along a single row or column, such as navigation bars and card content. CSS Grid is a two-dimensional system that controls both rows and columns simultaneously, making it best for page layouts and dashboards. Front-end developers often use Grid for major page structure and Flexbox for internal component alignment.",
                3: "CSS Flexbox is a one-dimensional layout model optimized for distributing space and aligning items along either a horizontal row or vertical column, making it content-driven and ideal for linear interface components like navigation bars, form controls, and card headers. In contrast, CSS Grid is a two-dimensional layout framework engineered to handle rows and columns simultaneously, making it layout-first and optimal for overarching page grids, dashboards, and magazine layouts. While Flexbox elements size dynamically based on their content and flex properties, Grid structures enforce rigid coordinate tracks using fractional units (fr) and grid templates. Modern front-end architectures compose both models collaboratively, utilizing Grid for page-level scaffolding and Flexbox for micro-component alignment."
            },
            # Q10: PWAs & Service Workers
            {
                1: "PWAs are web apps that can be installed and work offline using Service Workers to cache assets and manifests for metadata.",
                2: "Progressive Web Apps (PWAs) deliver app-like experiences in the browser with offline support and installability. The Service Worker is a background script that intercepts network requests and serves cached files using the Cache API when offline. The web app manifest file provides metadata like icons and display modes for home-screen installation.",
                3: "Progressive Web Apps (PWAs) are modern web applications engineered to deliver native application capabilities across browsers, including home-screen installability, offline availability, and background synchronization. At the architectural core of a PWA is the Service Worker, an independent background worker that intercepts incoming network requests and serves cached assets via the Cache API during network outages. The Web App Manifest is a JSON declaration that configures installation parameters, including app icons, splash screens, display mode (standalone), and theme colors. Together, these standards deliver resilient offline performance, fast load times, and push notifications without requiring distribution through proprietary app stores."
            }
        ],
        "hard": [
            # Q11: XSS and CSRF Defenses
            {
                1: "XSS is when attackers inject malicious scripts, while CSRF tricks users into sending unauthorized requests, defended with CSP and tokens.",
                2: "Cross-Site Scripting (XSS) occurs when malicious JavaScript is injected into a web page, which is prevented by input sanitization, output encoding, and Content Security Policy (CSP). Cross-Site Request Forgery (CSRF) tricks an authenticated user into executing unauthorized actions, which is mitigated using anti-CSRF tokens and SameSite cookies.",
                3: "Cross-Site Scripting (XSS) occurs when an attacker injects malicious client-side JavaScript into a trusted application, compromising session tokens and user data. Robust XSS defenses require context-aware output encoding, adopting modern frameworks with automatic escaping, implementing strict Content Security Policy (CSP) headers, and storing authentication tokens in HttpOnly cookies to block programmatic script access. Cross-Site Request Forgery (CSRF) exploits ambient browser authentication by tricking an authenticated session into submitting unauthorized state-changing requests. CSRF protection relies on synchronizer CSRF tokens, custom request verification headers (such as X-Requested-With), and setting cookie flags with SameSite=Lax or SameSite=Strict."
            },
            # Q12: Browser Rendering & Critical Path
            {
                1: "The browser builds the DOM and CSSOM to create the Render Tree, runs layout and paint, and optimizes by deferring non-critical scripts.",
                2: "The browser rendering engine constructs the DOM from HTML and the CSSOM from CSS, combining them into a Render Tree of visible elements. It then performs Layout to calculate element geometry, Paint to rasterize pixels, and Compositing via the GPU. Optimizing the Critical Rendering Path involves minimizing render-blocking CSS, deferring scripts with async/defer, and avoiding layout thrashing.",
                3: "The browser rendering engine converts markup into screen pixels by parsing HTML into the Document Object Model (DOM) and CSS into the CSS Object Model (CSSOM), synthesizing them into a Render Tree of visible nodes. The engine then initiates Layout (reflow) to calculate geometric dimensions, followed by Paint to rasterize pixels, and Compositing to assemble distinct layers onto the viewport using the GPU. Optimizing the Critical Rendering Path requires eliminating render-blocking resources by deferring non-essential JavaScript via async or defer, inlining critical above-the-fold CSS, and lazy-loading non-critical stylesheets. Furthermore, avoiding layout thrashing—caused by querying geometry properties immediately after mutating DOM nodes—and restricting animations to GPU-accelerated transforms and opacity preserves smooth 60fps rendering."
            },
            # Q13: Virtual DOM & React Reconciliation
            {
                1: "React uses a Virtual DOM and an O(n) diffing algorithm with stable keys to efficiently update only modified elements in the real DOM.",
                2: "The Virtual DOM is a lightweight memory representation of the actual DOM that React uses to optimize browser updates. When state changes, a new Virtual DOM tree is generated and compared against the previous tree using an O(n) heuristic diffing algorithm. Unique and stable keys in lists help the reconciliation process identify which items changed, were added, or were removed.",
                3: "The Virtual DOM is an in-memory JavaScript tree representing the actual browser DOM that UI frameworks utilize to batch and optimize browser updates. When component state changes, a new Virtual DOM tree is generated, and a heuristic reconciliation diffing algorithm compares it against the prior tree in linear O(n) time rather than O(n^3) brute-force tree comparison. The algorithm assumes elements of differing types produce entirely different subtrees and relies on unique, stable key props to track identity across dynamic list collections. Providing persistent keys enables the diffing engine to accurately isolate insertions, moves, and deletions, eliminating redundant DOM re-renders and preventing erroneous state destruction."
            },
            # Q14: WebSockets & WebRTC vs HTTP/2
            {
                1: "WebSockets offer full-duplex two-way communication, WebRTC provides peer-to-peer streaming, and SSE provides one-way server streaming.",
                2: "WebSockets provide persistent, full-duplex TCP connections over an upgraded HTTP handshake, ideal for real-time bidirectional messaging like chat. Server-Sent Events (SSE) offer lightweight unidirectional streaming from server to client over standard HTTP. WebRTC provides low-latency peer-to-peer audio, video, and data streaming over UDP, which is optimal for video calls.",
                3: "WebSockets provide persistent, bi-directional, full-duplex TCP connections initiated through an HTTP handshake upgrade, enabling low-latency data streaming where both client and server can transmit messages independently without request-response overhead. Server-Sent Events (SSE) offer unidirectional server-to-client streaming over standard HTTP, making them simpler and better suited for scenarios like live stock feeds or notifications that require no client-to-server pushing. WebRTC enables direct peer-to-peer audio, video, and arbitrary binary data streaming over UDP, minimizing latency for video conferencing and interactive multiplayer gaming. While HTTP/2 multiplexes multiple concurrent requests over a single connection, it remains request-response based, making WebSockets or WebRTC the preferred choice for real-time bidirectional communication."
            },
            # Q15: OAuth 2.0 with PKCE & OIDC
            {
                1: "OAuth 2.0 with PKCE protects Single Page Apps by exchanging code challenges instead of client secrets, and OIDC provides user identity tokens.",
                2: "Because Single Page Apps cannot securely store client secrets, OAuth 2.0 mandates the Authorization Code Flow with PKCE (Proof Key for Code Exchange) to prevent authorization code interception. OpenID Connect (OIDC) adds an ID token (JWT) to provide user identity claims alongside access tokens. For security, tokens should be stored in secure HttpOnly cookies rather than localStorage.",
                3: "For public Single Page Applications that cannot securely store client secrets, OAuth 2.0 mandates the Authorization Code Flow with Proof Key for Code Exchange (PKCE) to prevent code interception attacks. The client generates a cryptographic code verifier and derives a code challenge, sending the challenge with the initial authorization request and the verifier during token exchange to prove client authenticity. OpenID Connect (OIDC) extends this flow by issuing a cryptographically signed ID Token (JWT) that provides verifiable identity claims alongside the OAuth access token. In modern SPAs, storing access and refresh tokens in localStorage is avoided due to XSS vulnerabilities; instead, tokens should be handled via secure backend-for-frontend (BFF) proxies or stored in HttpOnly, SameSite=Strict, Secure cookies with rotating refresh tokens."
            }
        ]
    }
}


# ==============================================================================
# Dataset Generation Core Functions
# ==============================================================================

def load_question_bank(filepath: str) -> Dict[str, Any]:
    """
    Load interview questions from JSON file.

    Parameters:
    -----------
    filepath : str
        Absolute or relative path to question_bank.json.

    Returns:
    --------
    dict
        Structured question bank organized by role and difficulty level.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Question bank not found at path: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def generate_answers_for_question(
    role: str,
    difficulty: str,
    q_idx: int,
    question_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate 4 synthetic candidate answers (Labels 0, 1, 2, 3) for a single question.

    Quality Rubric:
    ---------------
    - Label 0 (Poor): 1-5 words, nervous, evasive, or 'I don't know'.
    - Label 1 (Needs Improvement): Vague, 1-sentence, lacks key concept depth.
    - Label 2 (Good): 2-3 sentences, accurate, mentions key concepts.
    - Label 3 (Excellent): 3-5 sentences, comprehensive, edge cases, senior-level.

    Parameters:
    -----------
    role : str
        Candidate role ('python_developer', 'data_analyst', 'web_developer').
    difficulty : str
        Difficulty level ('easy', 'medium', 'hard').
    q_idx : int
        0-based index of the question within the (role, difficulty) list.
    question_data : dict
        Question entry containing 'question', 'key_concepts', and 'reference_answer'.

    Returns:
    --------
    List[dict]
        Four rows ready for dataset export: question, answer, label, role.
    """
    question_text: str = question_data["question"]

    # --------------------------------------------------------------------------
    # Label 0 (Poor): Short nervous unprepared candidate answer
    # --------------------------------------------------------------------------
    poor_ans: str = random.choice(POOR_ANSWER_TEMPLATES)

    # --------------------------------------------------------------------------
    # Retrieve tailored responses for Labels 1, 2, 3
    # --------------------------------------------------------------------------
    role_responses = TAILORED_RESPONSES.get(role, {}).get(difficulty, [])
    if q_idx < len(role_responses):
        resp_dict = role_responses[q_idx]
        base_l1 = resp_dict[1]
        base_l2 = resp_dict[2]
        base_l3 = resp_dict[3]
    else:
        # Fallback if indices ever expand
        base_l1 = f"It is related to {role.replace('_', ' ')} concepts, but I am not completely sure."
        base_l2 = f"This concept is used in {role.replace('_', ' ')} for managing data and execution flow."
        base_l3 = question_data.get("reference_answer", "")

    # Inject natural randomized sentence openers for linguistic diversity
    good_opener = random.choice(GOOD_OPENERS)
    excellent_opener = random.choice(EXCELLENT_OPENERS)

    # Label 1: Keep concise single sentence
    needs_imp_ans: str = base_l1

    # Label 2: Good 2-3 sentences with natural phrasing
    good_ans: str = f"{good_opener}{base_l2[0].lower()}{base_l2[1:]}"

    # Label 3: Excellent comprehensive 3-5 sentences with senior phrasing
    excellent_ans: str = f"{excellent_opener}{base_l3[0].lower()}{base_l3[1:]}"

    rows = [
        {"question": question_text, "answer": poor_ans, "label": 0, "role": role},
        {"question": question_text, "answer": needs_imp_ans, "label": 1, "role": role},
        {"question": question_text, "answer": good_ans, "label": 2, "role": role},
        {"question": question_text, "answer": excellent_ans, "label": 3, "role": role},
    ]

    return rows


def generate_full_dataset(
    question_bank_path: str = QUESTION_BANK_PATH,
    output_csv_path: str = OUTPUT_CSV_PATH,
    seed: int = RANDOM_SEED
) -> List[Dict[str, Any]]:
    """
    Orchestrate full synthetic dataset generation and write to CSV.

    Parameters:
    -----------
    question_bank_path : str
        Source JSON file path.
    output_csv_path : str
        Destination CSV file path.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    List[dict]
        All generated records.
    """
    random.seed(seed)
    question_bank = load_question_bank(question_bank_path)

    all_rows: List[Dict[str, Any]] = []

    for role, difficulties in question_bank.items():
        for diff_level, questions in difficulties.items():
            for q_idx, q_item in enumerate(questions):
                generated_records = generate_answers_for_question(
                    role=role,
                    difficulty=diff_level,
                    q_idx=q_idx,
                    question_data=q_item
                )
                all_rows.extend(generated_records)

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Write to CSV with clean standard library writer
    fieldnames = ["question", "answer", "label", "role"]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(all_rows)

    return all_rows


def print_dataset_statistics(rows: List[Dict[str, Any]]) -> None:
    """
    Compute and print descriptive statistics of the generated dataset.
    Essential for viva presentations to prove balance and data quality.
    """
    total_count = len(rows)
    print("=" * 70)
    print("  SmartHire Training Dataset Statistics")
    print("=" * 70)
    print(f"Total Rows Generated : {total_count}")
    print("-" * 70)

    # Label distribution
    label_counts: Dict[int, int] = {}
    label_lengths: Dict[int, List[int]] = {}
    label_word_counts: Dict[int, List[int]] = {}

    for r in rows:
        lbl = r["label"]
        txt = r["answer"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

        label_lengths.setdefault(lbl, []).append(len(txt))
        label_word_counts.setdefault(lbl, []).append(len(txt.split()))

    label_names = {
        0: "0 (Poor / Unprepared)",
        1: "1 (Needs Improvement / Vague)",
        2: "2 (Good / Solid)",
        3: "3 (Excellent / Senior)"
    }

    print("Distribution by Answer Quality Label:")
    for lbl in sorted(label_counts.keys()):
        cnt = label_counts[lbl]
        pct = (cnt / total_count) * 100
        avg_chars = sum(label_lengths[lbl]) / cnt
        avg_words = sum(label_word_counts[lbl]) / cnt
        print(f"  - Label {label_names[lbl]}: {cnt} rows ({pct:.1f}%) | "
              f"Avg Words: {avg_words:.1f} | Avg Chars: {avg_chars:.1f}")

    print("-" * 70)

    # Role distribution
    role_counts: Dict[str, int] = {}
    for r in rows:
        rl = r["role"]
        role_counts[rl] = role_counts.get(rl, 0) + 1

    print("Distribution by Role:")
    for rl, cnt in sorted(role_counts.items()):
        pct = (cnt / total_count) * 100
        print(f"  - {rl:<18}: {cnt} rows ({pct:.1f}%)")

    print("-" * 70)
    print("Sample Records per Quality Tier:")
    printed_labels = set()
    for r in rows:
        lbl = r["label"]
        if lbl not in printed_labels:
            print(f"\n[Label {lbl} Sample - Role: {r['role']}]")
            print(f"  Question : {r['question'][:75]}...")
            print(f"  Answer   : {r['answer']}")
            printed_labels.add(lbl)
        if len(printed_labels) == 4:
            break

    print("=" * 70)
    print(f"Dataset successfully exported to: {OUTPUT_CSV_PATH}")
    print("=" * 70)


# ==============================================================================
# Standalone Execution Entrypoint
# ==============================================================================
if __name__ == "__main__":
    print(f"Initializing SmartHire dataset generator with seed {RANDOM_SEED}...")
    dataset_rows = generate_full_dataset(
        question_bank_path=QUESTION_BANK_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        seed=RANDOM_SEED
    )
    print_dataset_statistics(dataset_rows)
