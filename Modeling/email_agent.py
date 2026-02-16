from configuration import *
from model_setup import *

class EmailAgent:
    """
    Email Agent (System Bot):
    - Uses a pre-configured Bot account to send emails.
    - Uses LLM to draft a personalized summary for the recipient.
    - Attaches the professional PDF report.
    """
    def __init__(self, bot_email, bot_app_password, llm):
        self.bot_email = bot_email
        self.bot_app_password = bot_app_password
        self.llm = llm
        
        # Connection Settings (Gmail Default)
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587

    # ---------------- Helper: Extract user name from email ----------------
    def _extract_name_from_email(self, email):
        """
        Converts 'john.doe123@example.com' -> 'John Doe'
        Removes digits for a cleaner professional name.
        """
        try:
            local_part = email.split('@')[0]
            local_part = re.sub(r'\d+', '', local_part)          
            name_parts = re.split(r'[._-]+', local_part)
            name_parts = [part.capitalize() for part in name_parts if part]
            return " ".join(name_parts) if name_parts else "User"
        except Exception:
            return "User"

    # ---------------- Draft email intro using LLM ----------------
    def _draft_narrative(self, ml_results, user_email):
        """
        Uses the LLM to write a polished, professional email intro 
        summarizing the model's success, suitable for executives.
        """
        user_name = self._extract_name_from_email(user_email)

        prompt_template = PromptTemplate(
            input_variables=["model_name", "score", "user_name"],
            template="""
    You are a professional Data Science Assistant writing a formal email to a client.

    Email Requirements:
    - Address the client by name: {user_name}
    - Clearly state that the data analysis process is complete
    - Highlight the winning model: {model_name} and its evaluation score: {score}
    - Provide a concise executive-style summary of model performance
    - Mention that a detailed PDF report is attached
    - Use professional and polished business language
    - Keep the email concise (3-4 sentences)
    - End the email with the signature: "Best regards, Data Science Assistant"

    Output only the email body text.
    """
        )

        chain = prompt_template | self.llm
        narrative = chain.invoke({
            "model_name": ml_results["best_model_name"],
            "score": f"{ml_results['best_score']:.4f}",
            "user_name": user_name
        })

        return narrative



    # ---------------- Send Email ----------------
    def send_report(self, user_recipient_email, pdf_path, ml_results):
        """
        Main method to send the email.
        Only the recipient email is required from the user.
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file {pdf_path} not found.")
            return

        # 1. Prepare the email message
        msg = MIMEMultipart()
        msg['From'] = self.bot_email
        msg['To'] = user_recipient_email
        msg['Subject'] = f"Analysis Complete: {ml_results['best_model_name']} Performance Report"

        # 2. Draft the AI Narrative Body
        intro_text = self._draft_narrative(ml_results, user_recipient_email)
        msg.attach(MIMEText(intro_text, 'plain'))

        # 3. Attach the PDF File
        try:
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f"attachment; filename= {os.path.basename(pdf_path)}",
            )
            msg.attach(part)
        except Exception as e:
            print(f"Error attaching PDF: {e}")
            return

        # 4. Connect to SMTP Server and Send
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()  
            server.login(self.bot_email, self.bot_app_password)
            server.send_message(msg)
            server.quit()
            print(f"Success: Report sent to {user_recipient_email}")
        except Exception as e:
            print(f"SMTP Error: {e}")

BOT_MAIL = "ahmedismailan3@gmail.com"
BOT_PASS = os.getenv("BOT_GMAIL_PASS")

email_agent = EmailAgent(
    bot_email=BOT_MAIL,
    bot_app_password=BOT_PASS,
    llm=llm
)