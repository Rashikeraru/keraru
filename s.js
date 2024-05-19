const correctAnswers = {
    q1: 'a', // Engineering Math
    q2: 'b', // Network Security
    q3: 'a', // Database Management
    q4: 'a', // Database Management
    q5: 'a', // Computer Architecture
    q6: 'a', // Engineering Math
    q7: 'b', // Network Security
    q8: 'a', // Database Management
    q9: 'a', // Network Security
    q10: 'b', // Database Management
    q11: 'a', // Computer Architecture
    q12: 'c', // Database Management
    q13: 'a', // Network Security
    q14: 'b', // Computer Architecture
    q15: 'b', // Computer Architecture
};

function submitQuiz() {
    const form = document.getElementById('quizForm');
    const resultMessage = document.getElementById('resultMessage');
    const certificate = document.getElementById('certificate');
    const improvementMessage = document.getElementById('improvementMessage');
    const correctAnswersContainer = document.getElementById('correctAnswers');

    let score = 0;
    let totalQuestions = Object.keys(correctAnswers).length;
    let wrongAnswers = [];

    for (let question in correctAnswers) {
        const userAnswer = form[question].value;
        if (userAnswer === correctAnswers[question]) {
            score++;
        } else {
            wrongAnswers.push({ question, correctAnswer: correctAnswers[question] });
        }
    }

    const percentage = (score / totalQuestions) * 100;
    resultMessage.innerText = `Your score is ${percentage}%`;

    if (percentage >= 60) {
        certificate.innerHTML = '';
        certificate.style.display = 'block';
        improvementMessage.style.display = 'none';

        // Display certificate form for user to fill out details
        const certificateForm = `
            <h1>Generate Certificate</h1>
            <form id="certificateForm">
                <label for="studentName">Student Name:</label>
                <input type="text" id="studentName" name="studentName" required>
                <label for="courseName">Course Name:</label>
                <input type="text" id="courseName" name="courseName" required>
                <button type="button" onclick="generateCertificate()">Generate Certificate</button>
            </form>
        `;
        certificate.innerHTML = certificateForm;
    } else {
        certificate.style.display = 'none';

        let subjectsToImprove = wrongAnswers.map(answer => {
            switch (answer.question.charAt(1)) {
                case '1':
                case '6':
                    return 'Engineering Math';
                case '2':
                case '7':
                case '9':
                case '13':
                    return 'Network Security';
                case '3':
                case '4':
                case '8':
                case '10':
                case '12':
                    return 'Database Management';
                case '5':
                case '11':
                case '14':
                case '15':
                    return 'Computer Architecture';
                default:
                    return '';
            }
        });

        improvementMessage.innerText = `You need to improve in the following subjects: ${[...new Set(subjectsToImprove)].join(', ')}`;
        improvementMessage.style.display = 'block';
    }

    correctAnswersContainer.innerHTML = '<h3>Correct Answers:</h3>';
    wrongAnswers.forEach(answer => {
        const questionNumber = answer.question.slice(1);
        correctAnswersContainer.innerHTML += `<p>Question ${questionNumber}: ${answer.correctAnswer}</p>`;
    });

    correctAnswersContainer.style.display = 'block';
}

function generateCertificate() {
    const studentName = document.getElementById('studentName').value;
    const courseName = document.getElementById('courseName').value;

    if (studentName && courseName) {
        const certificateDiv = document.getElementById('certificate');
        certificateDiv.innerHTML = `
            <h2>Certificate of Completion</h2>
            <p>This certifies that</p>
            <h3>${studentName}</h3>
            <p>has successfully completed the course</p>
            <h3>${courseName}</h3>
        `;
        certificateDiv.style.display = 'block';
    } else {
        alert('Please enter both the student name and course name.');
    }
}
