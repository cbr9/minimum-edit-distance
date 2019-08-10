#!/usr/bin/python3

import numpy as np
from pandas import DataFrame
from typing import Tuple


class MED:
    
    def __init__(self):
        self.source = input("Source string: ")
        self.target = input("Target string: ")
        self.med_matrix = self.minimum_edit_distance()
        self.alignment = self.Alignment(match=1, mismatch=-1, gap=-1, 
                                        source=self.source, target=self.target)
    
    def __str__(self):
        return f'Minimum Edit Distance:\n' \
            f'{self.med_matrix}\n\n' \
            f"Minimum Edit Distance from '{self.source}' to '{self.target}' is {self.med_matrix.iloc[-1, -1]} according to Levenshtein's metric.\n" \
            f'{self.alignment.string_conversion}'
    
    def minimum_edit_distance(self, insert: int = 1, delete: int = 1, substitute: int = 2) -> DataFrame:
        
        source_chars = ["#"] + [char for char in self.source.replace(" ", "_")]
        target_chars = ["#"] + [char for char in self.target.replace(" ", "_")]
        
        med_matrix = np.zeros(shape=(len(self.source) + 1, len(self.target) + 1), dtype=int)
        med_matrix = DataFrame(data=med_matrix, index=source_chars, columns=target_chars)
        m, n = len(self.source), len(self.target)
        
        # initialize matrix
        for i in range(m + 1):
            med_matrix.iloc[i, 0] = i
        for j in range(n + 1):
            med_matrix.iloc[0, j] = j
        
        # fill up matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                med_matrix.iloc[i, j] = min(med_matrix.iloc[i - 1, j] + delete,
                                            med_matrix.iloc[i, j - 1] + insert,
                                            med_matrix.iloc[i - 1, j - 1] + substitute if self.source[i - 1] != self.target[j - 1]
                                            else med_matrix.iloc[i - 1, j - 1] + 0)
        return med_matrix
    
    class Alignment:
        
        def __init__(self, match: int, mismatch: int, gap: int, source: str, target: str, gap_filler: str = "-"):
            self.source = source
            self.target = target
            self.score_matrix = self.needleman_wunsch(match=match, mismatch=mismatch, gap=gap)
            self.aligned = self.alignment(match=match, mismatch=mismatch, gap=gap, gap_filler=gap_filler)
            self.string_conversion = self.convert_string(gap_filler=gap_filler)
        
        def needleman_wunsch(self, match: int, mismatch: int, gap: int) -> DataFrame:
            
            source_chars = ["#"] + [char for char in self.source.replace(" ", "_")]
            target_chars = ["#"] + [char for char in self.target.replace(" ", "_")]

            score_matrix = np.zeros(shape=(len(self.source) + 1, len(self.target) + 1), dtype=int)
            score_matrix = DataFrame(data=score_matrix, index=source_chars, columns=target_chars)
            m, n = len(self.source), len(self.target)
            
            # initialize matrix
            for i in range(m + 1):
                score_matrix.iloc[i, 0] = i * gap
            for j in range(n + 1):
                score_matrix.iloc[0, j] = j * gap
            
            # fill up matrix
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    score_matrix.iloc[i, j] = max(score_matrix.iloc[i - 1, j] + gap,
                                                  score_matrix.iloc[i, j - 1] + gap,
                                                  score_matrix.iloc[i - 1, j - 1] + match if self.source[i - 1] == self.target[j - 1]
                                                  else score_matrix.iloc[i - 1, j - 1] + mismatch)
            
            return score_matrix
        
        def alignment(self, match: int, mismatch: int, gap: int, gap_filler: str) -> Tuple[str, str]:
            
            i, j = len(self.source), len(self.target)
            
            source_aligned = ""
            target_aligned = ""
            
            # backtrace and alignment
            while i > 0 or j > 0:
                diagonal = mismatch if self.source[i - 1] != self.target[j - 1] else match
                if i > 0 and j > 0 and self.score_matrix.iloc[i, j] == self.score_matrix.iloc[i - 1, j - 1] + diagonal:
                    source_aligned = self.source[i - 1] + source_aligned
                    target_aligned = self.target[j - 1] + target_aligned
                    i -= 1
                    j -= 1
                elif i > 0 and self.score_matrix.iloc[i, j] == self.score_matrix.iloc[i - 1, j] + gap:
                    source_aligned = self.source[i - 1] + source_aligned
                    target_aligned = gap_filler + target_aligned
                    i -= 1
                else:
                    source_aligned = gap_filler + source_aligned
                    target_aligned = self.target[j - 1] + target_aligned
                    j -= 1
            
            return source_aligned, target_aligned
        
        def convert_string(self, gap_filler: str) -> str:
            
            source_aligned = [char for char in self.aligned[0]]
            target_aligned = self.aligned[1]
            
            conversion = ""
            conversion += "\nALIGNMENT:\n"
            conversion += f"Origin:   {''.join(source_aligned):10}\n"
            conversion += f"Target:  {target_aligned:10}\n\n"
            conversion += f"{'Step 0:':<10} {''.join(source_aligned).replace('-', ''):<15}\n"
            
            for i in range(len(source_aligned)):
                letter_src, letter_tgt = source_aligned[i], target_aligned[i]
                step = f"Step {str(i + 1)}:"
                origin = source_aligned.copy()
                origin = "".join(origin).replace(gap_filler, "")
                if letter_src == letter_tgt:
                    op = ""
                elif letter_src == gap_filler:
                    op = "(ins)"
                elif letter_tgt == gap_filler:
                    op = "(del)"
                else:
                    op = "(sub)"
                source_aligned[i] = target_aligned[i]
                changed = ''.join(source_aligned).replace(gap_filler, "")
                conversion += f"{step:<10} {origin:<15} >>> {changed:>15} {op:>15}\n"
            
            return conversion


if __name__ == "__main__":
    med = MED()
    print(med)
