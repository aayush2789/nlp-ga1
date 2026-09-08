"""Probabilistic Context-Free Grammar (PCFG) Induction and CKY/Viterbi Parser.

Induces a PCFG from the Penn Treebank sample and implements a robust,
most-probable-parse CKY algorithm with POS-tagset reconciliation and graceful
failure handling ('unparseable').
"""

from collections import defaultdict
from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Set, Tuple
import nltk
from nltk import Nonterminal, PCFG, induce_pcfg
from models.tagset_reconciler import default_reconciler


@dataclass
class ParseResult:
    """Encapsulates the result of a constituency parse."""
    is_valid: bool
    status: str  # 'parsed' or 'unparseable'
    log_prob: float
    tree_str: Optional[str] = None
    tree_nltk: Optional[nltk.Tree] = None
    latency_ms: float = 0.0


class ProbabilisticCKYParser:
    """Probabilistic CKY (Viterbi) constituency parser with tag smoothing."""

    def __init__(self, grammar: PCFG):
        self.grammar = grammar
        self.start = grammar.start()

        # Index grammar productions for fast bottom-up CKY lookups
        # rules_by_left[B][C] -> list of (A, log_prob)
        self.rules_by_left: Dict[Nonterminal, Dict[Nonterminal, List[Tuple[Nonterminal, float]]]] = defaultdict(lambda: defaultdict(list))
        # unary_rules[B] -> list of (A, log_prob)
        self.unary_rules: Dict[Nonterminal, List[Tuple[Nonterminal, float]]] = defaultdict(list)
        # lexical_rules[word.lower()] -> list of (Tag, log_prob)
        self.lexical_rules: Dict[str, List[Tuple[Nonterminal, float]]] = defaultdict(list)
        self.all_nonterminals: Set[Nonterminal] = set()
        self.cell_beam_k: int = 25  # Max nonterminals per chart cell

        self._index_grammar()

    def _index_grammar(self) -> None:
        """Partition productions into binary, unary, and lexical rule tables."""
        for prod in self.grammar.productions():
            lhs = prod.lhs()
            rhs = prod.rhs()
            prob = max(1e-12, prod.prob())
            log_p = math.log2(prob)
            self.all_nonterminals.add(lhs)

            if len(rhs) == 2:
                # Binary rule: A -> B C
                b_sym, c_sym = rhs[0], rhs[1]
                self.rules_by_left[b_sym][c_sym].append((lhs, log_p))
            elif len(rhs) == 1:
                rhs_sym = rhs[0]
                if isinstance(rhs_sym, Nonterminal):
                    # Unary nonterminal rule: A -> B
                    self.unary_rules[rhs_sym].append((lhs, log_p))
                else:
                    # Lexical terminal rule: Tag -> word
                    self.lexical_rules[str(rhs_sym).lower()].append((lhs, log_p))

    def _apply_unary_rules(
        self,
        chart_cell: Dict[Nonterminal, Tuple[float, any]],
        max_depth: int = 3,
    ) -> None:
        """Apply unary non-terminal closure to a single chart cell."""
        changed = True
        depth = 0
        while changed and depth < max_depth:
            changed = False
            depth += 1
            current_items = list(chart_cell.items())
            for rhs_sym, (rhs_log_p, rhs_bp) in current_items:
                if rhs_sym in self.unary_rules:
                    for lhs_sym, rule_log_p in self.unary_rules[rhs_sym]:
                        candidate_log_p = rule_log_p + rhs_log_p
                        if lhs_sym not in chart_cell or candidate_log_p > chart_cell[lhs_sym][0]:
                            chart_cell[lhs_sym] = (candidate_log_p, ("unary", lhs_sym, rhs_sym, rhs_bp))
                            changed = True

    def _prune_cell(self, cell: Dict[Nonterminal, Tuple[float, any]]) -> None:
        """Prune cell to top K most probable nonterminals to enforce linear-time CKY."""
        if len(cell) > self.cell_beam_k:
            top_items = sorted(cell.items(), key=lambda x: x[1][0], reverse=True)[: self.cell_beam_k]
            cell.clear()
            cell.update(top_items)

    def parse(
        self,
        tagged_words: List[Tuple[str, str]],
        max_sentence_len: int = 25,
    ) -> ParseResult:
        """Execute probabilistic CKY parsing on a sentence of (word, tag) pairs."""
        start_time = time.perf_counter()

        if not tagged_words:
            return ParseResult(
                is_valid=False, status="unparseable", log_prob=-float("inf"), latency_ms=0.0
            )

        n = len(tagged_words)
        if n > max_sentence_len:
            latency = (time.perf_counter() - start_time) * 1000.0
            return ParseResult(
                is_valid=False,
                status="unparseable",
                log_prob=-float("inf"),
                latency_ms=latency,
            )

        # 1. Tagset reconciliation: reconcile Q1 Brown tags to Penn Treebank tags
        reconciled_tagged = default_reconciler.reconcile_tagged_sentence(tagged_words)

        # 2. Initialize CKY chart
        chart: List[List[Dict[Nonterminal, Tuple[float, any]]]] = [
            [dict() for _ in range(n + 1)] for _ in range(n + 1)
        ]

        # 3. Fill diagonal cells (length 1 spans)
        for i in range(n):
            word, ptb_tag = reconciled_tagged[i]
            word_lower = word.lower()
            cell = chart[i][i + 1]

            matched_lexical = False
            if word_lower in self.lexical_rules:
                for tag_nt, log_p in self.lexical_rules[word_lower]:
                    cell[tag_nt] = (log_p, ("leaf", tag_nt, word))
                    matched_lexical = True

            ptb_nt = Nonterminal(ptb_tag)
            if not matched_lexical or ptb_nt not in cell:
                smoothed_log_p = -8.0
                cell[ptb_nt] = (smoothed_log_p, ("leaf", ptb_nt, word))

            self._apply_unary_rules(cell)
            self._prune_cell(cell)

        # 4. Dynamic programming loop over span lengths 2 to n
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                cell_ij = chart[i][j]

                for k in range(i + 1, j):
                    cell_ik = chart[i][k]
                    cell_kj = chart[k][j]

                    if not cell_ik or not cell_kj:
                        continue

                    # Fast intersection: only check (B, C) pairs where B -> C exists in grammar
                    for b_nt, (b_log_p, b_bp) in cell_ik.items():
                        if b_nt in self.rules_by_left:
                            b_children = self.rules_by_left[b_nt]
                            for c_nt in b_children:
                                if c_nt in cell_kj:
                                    c_log_p, c_bp = cell_kj[c_nt]
                                    for a_nt, rule_log_p in b_children[c_nt]:
                                        cand_log_p = rule_log_p + b_log_p + c_log_p
                                        if a_nt not in cell_ij or cand_log_p > cell_ij[a_nt][0]:
                                            cell_ij[a_nt] = (cand_log_p, ("binary", a_nt, b_bp, c_bp))

                self._apply_unary_rules(cell_ij)
                self._prune_cell(cell_ij)

        # 5. Extract best root parse over span [0, n]
        top_cell = chart[0][n]
        latency = (time.perf_counter() - start_time) * 1000.0

        if not top_cell:
            return ParseResult(
                is_valid=False,
                status="unparseable",
                log_prob=-float("inf"),
                latency_ms=latency,
            )

        # Prefer start symbol S, otherwise highest probability valid nonterminal
        best_root = None
        best_log_p = -float("inf")

        if self.start in top_cell:
            best_root = self.start
            best_log_p = top_cell[self.start][0]
        else:
            # Fallback root (e.g. S, SQ, SINV, NP, FRAG)
            for nt, (log_p, _) in top_cell.items():
                if log_p > best_log_p:
                    best_log_p = log_p
                    best_root = nt

        if best_root is None:
            return ParseResult(
                is_valid=False,
                status="unparseable",
                log_prob=-float("inf"),
                latency_ms=latency,
            )

        # 6. Reconstruct tree from backpointers
        best_bp = top_cell[best_root][1]
        tree_nltk = self._build_tree(best_bp)
        tree_str = str(tree_nltk)

        return ParseResult(
            is_valid=True,
            status="parsed",
            log_prob=best_log_p,
            tree_str=tree_str,
            tree_nltk=tree_nltk,
            latency_ms=latency,
        )

    def _build_tree(self, bp: any) -> nltk.Tree:
        """Recursively build an nltk.Tree from CKY backpointers."""
        kind = bp[0]
        if kind == "leaf":
            _, tag_nt, word = bp
            return nltk.Tree(str(tag_nt), [word])
        elif kind == "unary":
            _, lhs_nt, _, child_bp = bp
            return nltk.Tree(str(lhs_nt), [self._build_tree(child_bp)])
        elif kind == "binary":
            _, lhs_nt, left_bp, right_bp = bp
            return nltk.Tree(
                str(lhs_nt),
                [self._build_tree(left_bp), self._build_tree(right_bp)],
            )
        return nltk.Tree("X", [])


class PCFGManager:
    """Manages PCFG induction, caching, and parser instantiation."""

    _instance: Optional["PCFGManager"] = None

    def __init__(self, sample_size: int = 3500):
        self.sample_size = sample_size
        self.grammar = self._induce_grammar()
        self.parser = ProbabilisticCKYParser(self.grammar)

    def _induce_grammar(self) -> PCFG:
        """Induce PCFG in Chomsky Normal Form from Penn Treebank sample."""
        from data.corpora_loader import get_treebank_parsed_sents

        trees = get_treebank_parsed_sents(max_trees=self.sample_size)
        productions = []
        for t in trees:
            # Convert tree copy to CNF
            t_copy = t.copy(deep=True)
            t_copy.chomsky_normal_form()
            productions.extend(t_copy.productions())

        start_symbol = Nonterminal("S")
        grammar = induce_pcfg(start_symbol, productions)
        return grammar

    @classmethod
    def get_instance(cls, sample_size: int = 3500) -> "PCFGManager":
        if cls._instance is None:
            cls._instance = cls(sample_size=sample_size)
        return cls._instance
