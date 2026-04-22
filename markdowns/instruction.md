https://computational-physics.tripos.org/

#  **Due: 04/05/2026, 16:00** 


# Projects
The Computational Physics project — if you choose to submit one — is worth one unit of further work, so roughly 10% of your final mark for the year. It involves choosing a problem from the project list in Section 4. You will analyse the problem, write and test Python code to investigate it, then write up your work in a report. Like E1 and E2, you can expect it to involve 40 to 50 hours’ work. This includes reading and research, coding, experimentation and gathering results, and writing your report.

Students may start their project work once the project list is published by 20th February. Click on the GitHub Classroom link to accept the project as an assignment. The deadline for submission of the project report is 16:00 on the first Monday of Full Easter term (4th May 2026). Submissions made after that time will not be marked (see Section 2.1). I suggest you proceed as if the deadline is 16.00 on Friday 1st May, and use the final weekend if you really need to.

Bear in mind that everything in your report should be your own work, and your submission will be treated as a declaration of this fact. The rules regarding cheating and plagiarism in the Physics course handbook (Section 6) apply here. It is OK for you to use code that others — that’s what a library is, after all — but in all cases the attribution should be clear.

Notwithstanding the above, please also use the discussions to ask each other questions and share knowledge.





# 1 The report
Your report will take the form of repository submitted to GitHub classroom (see Section 2). It should consist of a Jupyter notebook containing the body of the report, with all plots rendered (you can check how it will look on GitHub), together with any additional code written for the project as modules in .py files, and any data that you have produced in the course of running your code (e.g. simulation data in the form of .npy or .npz files) that is analyzed further in your report.

When the cells in the notebook are executed, all analysis and figures should be reproduced quickly. Execution should not cause long runs of simulation code, for example. While you should feel free to use other (open source) libraries in your code, please limit your report notebook to the standard libraries we have used in the computational physics course (NumPy, SciPy, Matplotlib).

Please see Keeping Laboratory Notes and Writing Formal Reports for further guidance. There is no prescription on the length of your report, except that it should provide a comprehensive account of the work you have done.

For a guide to the presentation of Python code, see PEP 8 and this explanation.





# 2 Submission
The submission instructions are the same as for the exercises. Click on the GitHub Classroom link to accept the project as an assignment. Please change the name of your repo to <project-name>-<CRSid> for ease of marking.

You should take advantage of the fact that you are using GitHub to regularly commit to your project repository and push the commits to GitHub. That way there is no danger of you accidentally missing the deadline: your last commit to the repository on GitHub will constitute your submission.

2.1 Late submission and extensions
See the NST rules on late submission: “Students are required to self-certify before a submission deadline or at the point of submission of the work. Retrospective requests will not be accepted. Where a student does not self-certify and does not submit their coursework by the original submission date, zero marks will be awarded. Where a student has self-certified and does not submit their work by the revised submission date, zero marks will be awarded, unless a further extension has been granted by the department or EAMC”. The form for submitting a self-certified extension is here. Do not contact me directly with extension requests.





# 3 Marking
The credit for the project is one unit of further work (like TP1, TP2, E1, E2, etc.). The projects are marked out of ten in the four categories:

Analysis of the computational physics aspects of the problem (possible algorithms, their complexity, etc.).
Details of implementation of the algorithm and its performance (e.g. in terms of run time on what hardware). This will include the description in the report as well as the accompanying code, and will include a judgement of style, readability (including comments and docstrings) and quality.
Results, analysis of errors (if applicable), tests and discussion of the relevant computational physics.
Overall presentation of the report, structure, etc.
Although presentation only enters in the last category, bear in mind that assessors will not be able to get a good sense of what you have done unless your work is presented clearly throughout. In particular, it will make your assessor’s life easier (always a good thing) if there are sections that clearly refer to the first three points.

As a general guide, you should present your work as if your assessor is completely ignorant of both the problem and computational approaches to it. Alternatively, present your work as if to yourself before you began working on it.




# 4 The projects
For all of these projects the basic protocol is the same:

Understand the physical problem
Learn about the available algorithms, including their theoretical complexity / performance
Implement the algorithm(s)
Gather results about the performance of the method
You will need to do some research and reading before starting, so follow the links and references1 given, or find your own. But read pragmatically: look for the algorithm description or pseudo-code and start thinking about how you will translate it into your own code.





# 4.2 Hierarchical methods for $N$-body simulation
The [Fast Multipole method](https://en.wikipedia.org/wiki/Fast_multipole_method) and the [Barnes Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) method are two methods to study pairwise interactions in N-body systems that lower the complexity to $n\log{n}$
from the naive $n^{2}$. 
Implement these methods and compare their performance on different tasks.



