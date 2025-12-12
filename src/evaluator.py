
"""
Single round of evaluation for a given solution.

Example usage:

    judge = FrontierCSEvaluator()
    result = judge.evaluate_solution(
        problem_track="algorithmic",
        problem_id=108,
        solution_code=\"\"\"
        #include <bits/stdc++.h>
        using namespace std;
        int main() {
            ios::sync_with_stdio(false);
            cin.tie(nullptr);

            int n, m;
            if (!(cin >> n >> m)) return 0;

            cout << "!";
            for (int i = 1; i < n; ++i) cout << " 0";
            cout << "\\n";
            cout.flush();
            return 0;
        }
        \"\"\"
    )
"""

import time
import requests

class FrontierCSEvaluator:
    def __init__(self, judge_url="http://localhost:8081"):
        self.judge_url = judge_url
        self.session = requests.Session()
    
    def submit_algorithmic_solution(self, pid, code):
        files = {'code': ('solution.cpp', code)}
        data = {'pid': pid, 'lang': 'cpp'}
        try:
            response = self.session.post(f"{self.judge_url}/submit", files=files, data=data)
            response.raise_for_status()
            return response.json().get('sid')
        except requests.RequestException:
            return None
    
    def get_result(self, sid, poll_interval=10):
        while True:
            try:
                response = self.session.get(f"{self.judge_url}/result/{sid}")
                if response.status_code == 404:
                    time.sleep(poll_interval)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if result.get('status') in ['done', 'error']:
                    return result
                
                time.sleep(poll_interval)
                
            except requests.RequestException:
                time.sleep(poll_interval)
    
    def evaluate_solution(self, problem_track: str, problem_id: int, solution_code: str) -> float:
        if problem_track == "algorithmic":
            sid = self.submit_algorithmic_solution(problem_id, solution_code)
            if not sid:
                return {'status': 'error', 'message': 'Submission failed'}
            result = self.get_result(sid)
            return result.get('score', 0.0)
        else:
            return {'status': 'error', 'message': f'Unknown problem track: {problem_track}'}