# Msc_Thesis_Demokritos_OPSYEDAI
# Οδηγός Χρήσης

<summary><strong style="font-size: 2em;">Εκπαίδευση LLM</strong></summary>
Για να γίνει εκπαίδευση του LLM, αρκεί να τρέξουμε το αρχείο <a href="notebooks/LLM training.ipynb">LLM training.ipynb</a>, στο οποίο διαβάζουμε και επεξεργαζόμαστε τα δεδομένα όπως τα περιμένουν τα Llama-based LLMs, φορτώνουμε και εκπαιδεύουμε το LLM που θέλουμε, και μετά το αποθηκεύουμε στον φάκελο <a href="saved_models">saved_models</a>. Αρκετή από τη λειτουργικότητα βρίσκεται πίσω από συναρτήσεις του φακέλου <a href="utils">utils</a>



<summary><strong style="font-size: 2em;">Εκπαίδευση IRMs</strong></summary>
Η εκπαίδευση των Information Retrieval μοντέλων γίνεται μέσω συναρτήσεων στη <a href="utils">utils</a>. Δεν αποθηκεύουμε κάποιο μοντέλο, καθώς η εκπαίδευση τους είναι γρήγορη και γίνεται αυτόματα κατά την αρχικοποίησή τους. Ένα παράδειγμα αρχικοποίησης είναι το εξής:

```python
from utils.qna_irm import create_corpus, build_qna_sklearn_index, qna_irm_pipeline
from utils.pdf_irm import split_pdf_by_sections_skip_intro, build_pdf_sklearn_index, pdf_irm_pipeline
from utils.datahandling import load_database

qna_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
pdf_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
qna_data_path = "../datasets/combined_dataset.json"
pdf_data_path = "../datasets/opsyed_hkely_user_manual.pdf"


# QnA IRM preparation
data = load_database(qna_data_path)
corpus, answers = create_corpus(data)
index, model, _ = build_qna_sklearn_index(corpus, qna_model_name)

# pdf IRM preparation
sections = split_pdf_by_sections_skip_intro(pdf_data_path)
chunks = [section['content'] for section in sections]
titles = [section['section'] for section in sections]

pdf_index, pdf_model, _ = build_pdf_sklearn_index(chunks, pdf_model_name)
```



<summary><strong style="font-size: 2em;">Τρέξιμο ChatBot</strong></summary>
Για να τρέξουμε το OpsyedAI chatbot πρέπει να τρέξουμε το αρχείο <a href=host_model.py>host_model.py</a> το οποίο αρχικοποιεί το μοντέλο και σηκώνει ένα API (FastAPI) για τη διεκπαιρέωση αιτημάτων στη συνάρτηση 'generate'. Μπορούμε μετά να ανοίξουμε σε ένα browser το αρχείο <a href=chatbot.html>chatbot.html</a> ώστε να κάνουμε αιτήματα στο API μέσω του γραφικού περιβάλλοντος της σελίδας αυτής


<hr>

# Περιγραφή φακέλων και αρχείων του repo


<details>
  <summary><strong style="font-size: 2em;">📁 <a href="utils">utils</a></strong></summary>
  <div style="margin-left: 20px;">
  Ο φάκελος αυτός εμπεριέχει αρχεία με ορισμούς συναρτήσεων που χρησιμοποιούνται από τα notebooks, καθώς και κάποια λειτουργικά αρχεία όπως το host του μοντέλου και μια ιστοσελίδα που σηκώνουμε τοπικά για πειραματισμό

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="utils/datahandling.py">datahandling.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Συναρτήσεις για φόρτωμα και επεξεργασία δεδομένων
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="utils/modeling.py">modeling.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Συναρτήσεις για το φόρτωμα και τις λειτουργίες του LLM μοντέλου. Το μοντέλο φορτώνεται με BitsAndBytes quantization με 4-bit ακρίβεια, ώστε να χωράει σε μικρότερες gpu για εκπαίδευση. Κατά την εκπαίδευση χρησιμοποιείται LoRA (Low-Rank Adaptation) ώστε να εκπαιδεύονται στοχευμένα βάρη και όχι όλο το προεκπαιδευμένο μοντέλο, το οποίο βοηθάει στην ταχύτητα εκπαίδευσης και στο κόστος μνήμης
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="utils/pdf_irm.py">pdf_irm.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Συναρτήσεις για τη λειτουργικότητα του pdf Iformation Retrieval Model (pdf IRM). Τα κεφάλαια του pdf χωρίζονται με regex που βρίσκει αριθμημένους τίτλους (π.χ. "2 Εγγραφή", "2.1 Σύνδεση", "5.3 Τα ραντεβού μου"), οπότε έχει και μια μικρή πιθανότητα σφάλματος. Έπειτα, τα περιεχόμενα των κεφαλαίων αναπαριστώνται χρησιμοποιώντας SentenceTransformer. Για κάποιο πιθανό κείμενο, μπορούμε να βρούμε το πιο σχετικό κεφάλαιο, εξετάζωντας την αναπαράσταση του κειμένου αυτού, με τις αναπαραστάσεις που φτιάξαμε ππρωτίστως, χρησιμοποιώντας NearestNeighbors. Υπάρχει επίσης η λειτουργία να απαντάμε πως η ερώτηση δεν είναι σχετική με τις λειτουργίες του μοντέλου μας, αν είναι πολύ μακριά από όλες τις ερωτήσεις στο QnA
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="utils/qna_irm.py">qna_irm.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Συναρτήσεις για τη λειτουργικότητα του QnA Iformation Retrieval Model (QnA IRM). Οι ερωτήσεις ενός QnA dataset αναπαριστώνται χρησιμοποιώντας SentenceTransformer. Για κάποια πιθανή ερώτηση, μπορούμε να βρούμε την πιο σχετική ερώτηση του QnA, εξετάζωντας την αναπαράσταση της ερώτησης αυτής, με τις αναπαραστάσεις που φτιάξαμε ππρωτίστως, χρησιμοποιώντας NearestNeighbors
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="utils/opsyed_pipeline.py">opsyed_pipeline.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Συναρτήσεις για την αρχικοποίηση και τη λειτουργικότητα του τελικού pipeline, που χρησιμοποιεί το LLM, και τα QnA IRM και pdf IRM
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="utils/analyze_metrics.py">analyze_metrics.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Κώδικας για την ανάλυση των σκορ και τη γραφική τους αναπαράσταση με pyplot. Η τωρινή υλοποίηση αναμένει ένα xlsx αρχείο με συγκεκριμένο μορφότυπο, ένα παράδειγμα μπορεί να βρεθεί στο <a href="results/ΗΚΕΛΥ.xlsx">ΗΚΕΛΥ.xlsx</a>
    </div>
  </details>
  </div>
</details>

<hr>

<details>
  <summary><strong style="font-size: 1.5em;">📄 <a href="utils/host_model.py">host_model.py</a></strong></summary>
  <div style="margin-left: 20px;">
  Κάνει host το μοντέλο τοπικά μέσω FastAPI. Τα requests μπορούν να γίνουν στη συνάρτηση 'generate' με είσοδο απλά ένα string (την ερώτηση του χρήστη), και επιστρέφονται η απάντηση του OpsyedAI, καθώς και κάποιες boolean τιμές για το αν ενεργοποιήθηκε το QnA IRM, το pdf IRM, και αν η ερώτηση ήταν σχετική
  </div>
</details>

<details>
  <summary><strong style="font-size: 1.5em;">📄 <a href="utils/chatbot.html">chatbot.html</a></strong></summary>
  <div style="margin-left: 20px;">
  Μια τοπική σελίδα περιβάλλοντος τεσταρίσματος του opsyedai chatbot που στέλνει requests στο μοντέλο που γίνεται host από το <a href="utils/host_model.py">host_model.py</a>
  </div>
</details>

<details>
  <summary><strong style="font-size: 1.5em;">📄 <a href="utils/load_tests.py">load_tests.py</a></strong></summary>
  <div style="margin-left: 20px;">
  Χρησιμοποιήθηκε για να δοκιμαστεί το pipeline μας σε μια προσομοίωση αληθινού περιβάλλοντος με πολλαπλούς χρήστες και πιθανότητα ταυτόχρονων requests. Στέλνει requests στο μοντέλο που γίνεται host από το <a href="utils/host_model.py">host_model.py</a>
  </div>
</details>

<hr>

<details>
  <summary><strong style="font-size: 2em;">📁 <a href="notebooks">notebooks</a></strong></summary>
  <div style="margin-left: 20px;">
  Αυτός ο φάκελος εμπεριέχει όλα τα πειράματα και τον κώδικα που χρησιμοποιήθηκε κατά την έρευνα που έγινε για αυτό το task

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="notebooks/LLM training.ipynb">LLM training.ipynb</a></strong></summary>
    <div style="margin-left: 20px;">
    Κώδικας για τη φόρτωση προεκπαιδευμένου LLM και περαιτέρω εκπαίδευση στα δικά μας δεδομένα. Το μοντέλο που χρησιμοποιήσαμε είναι το <a href=https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5>ilsp/Meltemi-7B-Instruct-v1.5</a>, λόγω της εξοικίωσής του με τα Ελληνικά
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="notebooks/QnA IRM.ipynb">QnA IRM.ipynb</a></strong></summary>
    <div style="margin-left: 20px;">
    Κώδικας για τη δοκιμή λειτουργικότητς και αποτελεσματικότητας (μέσω σκορ) του QnA IRM
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="notebooks/PDF IRM.ipynb">PDF IRM.ipynb</a></strong></summary>
    <div style="margin-left: 20px;">
    Κώδικας για τη δοκιμή λειτουργικότητς και αποτελεσματικότητας (μέσω πειραμάτων) του PDF IRM
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="notebooks/full pipeline.ipynb">full pipeline.ipynb</a></strong></summary>
    <div style="margin-left: 20px;">
    Κώδικας για τη δοκιμή λειτουργικότητας και αποτελεσματικότητας (μέσω πειραμάτων, και σκορ) ολόκληρου του pipeline (LLM, QnA IRM και PDF IRM)
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="notebooks/save_full_best.ipynb">save_full_best.ipynb</a></strong></summary>
    <div style="margin-left: 20px;">
    Όταν αποθηκεύουμε το μοντέλο, αποθηκεύεται ένα ποσοστό των παραμέτρων του, διότι χρησιμοποιούμε μια μέθοδο κατά την εκπαίδευση που τροποποιεί μόνο αυτό το ποσοστό των παραμέτρων (μέθοδος LoRA). Αυτό το αρχείο δίνει τη δυνατότητα να αποθηκεύσουμε το μοντέλο με όλες τις παραμέτρους του.
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="notebooks/fast_inference.ipynb">fast_inference.ipynb</a></strong></summary>
    <div style="margin-left: 20px;">
    Αυτό το αρχείο τρέχει το LLM και καταγράφει τους χρονισμούς αποκρίσεων του LLM ανά token. Το χρησιμοποιήσαμε για να δούμε πιθανές βελτιώσεις όταν τροποποιούσαμε κάτι στο LLM
    </div>
  </details>
  

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="notebooks/RLHF.ipynb">RLHF.ipynb</a></strong></summary>
    <div style="margin-left: 20px;">
    Κώδικας για τη δοκιμή λειτουργικότητας του Reinforcement Learning from Human Feedback (RHLF)
    </div>
  </details>
  </div>
</details>

<hr>

<details>
  <summary><strong style="font-size: 2em;">📁 <a href="saved_models">saved_models</a></strong></summary>
  <div style="margin-left: 20px;">
  Αυτός ο φάκελος χρησιμοποιείται για να αποθηκεύονται οι εκπαιδεύσεις μοντέλων
  </div>
</details>

<hr>

<details>
  <summary><strong style="font-size: 2em;">📁 <a href="datasets">datasets</a></strong></summary>
  <div style="margin-left: 20px;">
  Εμπεριέχει αρχεία με δεδομένα που μπορούν να χρησιμοποιηθούν για εκπαιδεύσεις, καταγραφή σκορ, και ανακτήσεις από Μοντέλα Ανάκτησης (Information Retrieval Models). Τα δεδομένα αφορούν κυρίως ερωτοαποκρίσεις επάνω στο περιβάλλον του ΗΚΕΛΥ

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="datasets/dataset_v2.json">dataset_v2.json</a></strong></summary>
    <div style="margin-left: 20px;">
    Δεδομένα 294 ερωτοαποκρίσεων, σχετικά με το ΗΚΕΛΥ, και γραμμένα από ανθρώπινο χέρι (όχι από χρήστες, από εμάς)
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="datasets/opsyed_hkely_user_manual.pdf">opsyed_hkely_user_manual.pdf</a></strong></summary>
    <div style="margin-left: 20px;">
    Εγχειρίδιο χρήσης Ηλεκτρονικού Κέντρου Εξυπηρέτησης Ληπτών Υγείας (Η.Κ.Ε.Λ.Υ). Χρησιμοποιείται στο PDF IRM μοντέλο για την εύρεση σχετικού κεφαλαίου με την ερώτηση του χρήστη
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="datasets/images.json">images.json</a></strong></summary>
    <div style="margin-left: 20px;">
    Metadata για τις φωτογραφίες που εμπεριέχονται στο <a href="datasets/opsyed_hkely_user_manual.pdf">opsyed_hkely_user_manual.pdf</a>
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="datasets/combined_dataset.json">combined_dataset.json</a></strong></summary>
    <div style="margin-left: 20px;">
    Συνδυασμός των αρχείων <a href="datasets/dataset_v2.json">dataset_v2.json</a> και <a href="datasets/images.json">images.json</a> όπου μια ερώτηση σχετίζεται με εικόνες
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="datasets/rephrased_faq_v1.json">rephrased_faq_v1.json</a></strong></summary>
    <div style="margin-left: 20px;">
    Αναδιατυπώσεις 40 ερωτήσεων από το <a href="datasets/combined_dataset.json">combined_dataset.json</a> για χρησιμοποίησή τους για μέτρηση απόδοσης του μοντέλου σε 'άγνωστα' κείμενεα
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="datasets/rephrased_faq_v2.json">rephrased_faq_v2.json</a></strong></summary>
    <div style="margin-left: 20px;">
    Διαφορετικές αναδιατυπώσεις των ίδιων 40 ερωτήσεων που αναδιατυπώνονται στο <a href="datasets/rephrased_faq_v1.json">rephrased_faq_v1.json</a>
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="datasets/rephrased_faq_v3.json">rephrased_faq_v3.json</a></strong></summary>
    <div style="margin-left: 20px;">
    Ακόμα άλλες διαφορετικές αναδιατυπώσεις των ίδιων 40 ερωτήσεων που αναδιατυπώνονται στο <a href="datasets/rephrased_faq_v1.json">rephrased_faq_v1.json</a>. Συνολικά τα 3 αρχεία έχουν 120 ερωτήσεις (3 αναδιατυπώσεις 40 ερωτήσεων από το <a href="datasets/combined_dataset.json">combined_dataset.json</a>)
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="datasets/test_ratings.csv">test_ratings.csv</a></strong></summary>
    <div style="margin-left: 20px;">
    Περιέχει το πρότυπο των δεδομένων που θα χρειαστούν για RHLF. Χρησιμοποιήθηκε και για να δοκιμαστεί ο κώδικας του RHLF
    </div>
  </details>
  </div>
</details>

<hr>

<details>
  <summary><strong style="font-size: 2em;">📁 <a href="instructions">instructions</a></strong></summary>
  <div style="margin-left: 20px;">
  Εμπεριέχει αρχεία με προκαθορισμένα κείμενα που μπορούν να δωθούν στο LLM ως οδηγίες ή να απαντάει το LLM με αυτά

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="instructions/v1 LLM instructions.txt">v1 LLM instructions.txt</a></strong></summary>
    <div style="margin-left: 20px;">
    Οι τωρινές οδηγίες που δίνονται στο μοντέλο. Λιτές και γραμμένες με το χέρι
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="instructions/v1 LLM unrelated response.txt">v1 LLM unrelated response.txt</a></strong></summary>
    <div style="margin-left: 20px;">
    Η τωρινή αυτοματοποιημένη απάντηση που δίνει το μοντέλο όταν θεωρεί πως η ερώτηση δεν ήταν σχετική
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="instructions/pdf suggestion.txt">pdf suggestion.txt</a></strong></summary>
    <div style="margin-left: 20px;">
    Το τωρινό αυτοματοποιημένο μήνυμα που προσθέτει το μοντέλο στην απάντησή του όταν θεωρεί πως η ερώτηση σχετίζεται με συγκεκριμένα κείμενα του <a href="datasets/guide.pdf">guide.pdf</a>
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="instructions/AI Agent Instructions.txt">AI Agent Instructions.txt</a></strong></summary>
    <div style="margin-left: 20px;">
    Υπεραναλυτικές οδηγίες που έδωσε το chatgpt όταν του περιγράψαμε το μοντέλο που έχουμε, τον σκοπό του, και αρκετούς από τους στόχους μας για αυτό. Είναι στα Αγγλικά, οπότε δε μας χρησιμεύει σε αυτή τη μορφή
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="instructions/AI Agent Instructions Greek.txt">AI Agent Instructions Greek.txt</a></strong></summary>
    <div style="margin-left: 20px;">
    Μετάφραση του chatgpt για το <a href="AI Agent Instructions.txt">AI Agent Instructions.txt</a> η οποία πέρασε και από ανθρώπινο χέρι για διορθώσεις και αλλαγές. Όταν δώθηκαν στο μοντέλο είχε χειρότερο σκορ από τις βασικές οδηγίες (αρχείο <a href="v1 LLM instructions.txt">v1 LLM instructions.txt</a>)
    </div>
  </details>
  </div>
</details>

<hr>

<details>
  <summary><strong style="font-size: 2em;">📁 <a href="legacy">legacy</a></strong></summary>
  <div style="margin-left: 20px;">

  Εμπεριέχει αρχεία από τις παλιές υλοποιήσεις που είχαν γίνει

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="legacy/meltemi.py">meltemi.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Αυτός ο κώδικας μετατρέπει ένα γενικό ελληνικό γλωσσικό μοντέλο (Meltemi-7B) σε έναν εξειδικευμένο βοηθό για ιατρικά ραντεβού. Συγκεκριμένα:
    <ul>
      <li>Φορτώνει ερωτήσεις και απαντήσεις σχετικά με ιατρικά ραντεβού από ένα αρχείο JSON</li>
      <li>Χρησιμοποιεί μια τεχνική που λέγεται LoRA για να εκπαιδεύσει το μοντέλο με αποδοτικό τρόπο, αλλάζοντας μόνο ένα μικρό μέρος των παραμέτρων του</li>
      <li>Εφαρμόζει κβαντοποίηση 4-bit για να μειώσει τις απαιτήσεις μνήμης, επιτρέποντας την εκπαίδευση ακόμα και σε υπολογιστές με περιορισμένη μνήμη GPU</li>
      <li>Μετά την εκπαίδευση, αποθηκεύει το προσαρμοσμένο μοντέλο και δημιουργεί ένα απλό περιβάλλον συνομιλίας όπου ο χρήστης μπορεί να κάνει ερωτήσει στα ελληνικά σχετικά με ιατρικά ραντεβού και να λάβει σχετικές απαντήσεις</li>
    </ul>
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="legacy/load_meltemi.py">load_meltemi.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Αυτός ο κώδικας φορτώνει και χρησιμοποιεί το ήδη εκπαιδευμένο μοντέλο του ιατρικού βοηθού για συνομιλία. Συγκεκριμένα:
    <ul>
      <li>Δημιουργεί έναν φάκελο για προσωρινή αποθήκευση τμημάτων του μοντέλου (offload) ώστε να βελτιστοποιηθεί η χρήση μνήμης</li>
      <li>Ρυθμίζει κβαντοποίηση 4-bit για να μειώσει τις απαιτήσεις μνήμης GPU κατά τη φόρτωση του μοντέλου</li>
      <li>Φορτώνει το προηγουμένως εκπαιδευμένο μοντέλο και τον tokenizer από τον φάκελο "saved_models/meltemi-greek-medical-assistant-final-instr"</li>
      <li>Ορίζει μια συνάρτηση generate_response που παίρνει ως είσοδο την ερώτηση του χρήστη, τη μορφοποιεί κατάλληλα με τις ετικέτες [INST], και επιστρέφει την απάντηση του μοντέλου</li>
      <li>Δημιουργεί ένα απλό διαδραστικό περιβάλλον συνομιλίας στην κονσόλα, όπου ο χρήστης μπορεί να υποβάλει ερωτήσεις σχετικά με ιατρικά ραντεβού και να λάβει απαντήσεις</li>
    </ul>
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="legacy/meltemi_feedback.py">meltemi_feedback.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Αυτός ο κώδικας δημιουργεί ένα διαδραστικό περιβάλλον για τον ιατρικό βοηθό ραντεβού με προσθήκη συστήματος ανατροφοδότησης για τη συνεχή βελτίωση του μοντέλου. Συγκεκριμένα: Φορτώνει το προεκπαιδευμένο μοντέλο του ιατρικού βοηθού με κβαντοποίηση 4-bit για βέλτιστη χρήση μνήμης. Δημιουργεί φακέλους για την προσωρινή αποθήκευση τμημάτων του μοντέλου και για την αποθήκευση της ανατροφοδότησης των χρηστών. Υλοποιεί ένα σύστημα συλλογής ανατροφοδότησης που:
    <ul>
      <li>Ρωτά τον χρήστη αν η απάντηση ήταν ικανοποιητική</li>
      <li>Σε περίπτωση αρνητικής απάντησης, ζητά από τον χρήστη να παρέχει τη σωστή απάντηση</li>
      <li>Αποθηκεύει όλη την ανατροφοδότηση (θετική και αρνητική) σε ξεχωριστά αρχεία JSON
      Παρέχει μια συνάρτηση (prepare_fine_tuning_dataset) που μπορεί να χρησιμοποιηθεί για τη δημιουργία νέου συνόλου δεδομένων εκπαίδευσης από την συλλεγμένη ανατροφοδότηση.
      Το τελικό αρχείο εκπαίδευσης περιλαμβάνει:</li>
      <ul>
        <li>Τις διορθωμένες απαντήσεις από τη αρνητική ανατροφοδότηση</li>
        <li>Τις αρχικές απαντήσεις του μοντέλου που έλαβαν θετική ανατροφοδότηση
        Ο κώδικας όχι μόνο επιτρέπει στους χρήστες να συνομιλήσουν με τον ιατρικό βοηθό, αλλά δημιουργεί και ένα σύστημα συνεχούς βελτίωσης μέσω της ανατροφοδότησης, επιτρέποντας στο μοντέλο να γίνεται όλο και πιο ακριβές με την πάροδο του χρόνου, καθώς μαθαίνει από τα λάθη του και ενισχύει τις σωστές απαντήσεις του</li>
      </ul>
    </ul>
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="legacy/terminal_for_meltemi.py">terminal_for_meltemi.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Αυτός ο κώδικας δημιουργεί μια προηγμένη διεπαφή τερματικού για τον ιατρικό βοηθό ραντεβού με οπτικές βελτιώσεις και εφέ. Συγκεκριμένα: Χρησιμοποιεί τη βιβλιοθήκη colorama για να προσθέσει χρώματα και στυλ στο κείμενο της κονσόλας, λειτουργώντας σωστά σε όλα τα λειτουργικά συστήματα. Προσθέτει οπτικά εφέ όπως:
    <ul>
      <li>Καθαρισμό οθόνης</li>
      <li>Στυλιζαρισμένη επικεφαλίδα με πλαίσιο</li>
      <li>Εφέ πληκτρολόγησης κατά την εμφάνιση των απαντήσεων</li>
      <li>Κινούμενο animation "σκέψης" κατά τη διάρκεια της δημιουργίας απάντησης</li>
      <li>Παρέχει λειτουργίες διεπαφής όπως:</li>
      <li>Εντολή "έξοδος" για τερματισμό</li>
      <li>Εντολή "καθαρισμός" για καθαρισμό της οθόνης</li>
      <li>Διαχείριση σφαλμάτων και διακοπών πληκτρολογίου
      <li>Διατηρεί το βασικό μοντέλο πίσω από τη διεπαφή:</li>
      <li>Φορτώνει το μοντέλο με κβαντοποίηση 4-bit για βέλτιστη απόδοση</li>
      <li>Δημιουργεί απαντήσεις με την ίδια ποιότητα
      Ο κώδικας προσθέτει μια επαγγελματική, φιλική προς τον χρήστη και οπτικά ελκυστική διεπαφή στον βοηθό ιατρικών ραντεβού, κάνοντας την αλληλεπίδραση πιο ευχάριστη και εύχρηστη, ενώ παράλληλα διατηρεί όλη τη λειτουργικότητα του μοντέλου</li>
    </ul>
    </div>
  </details>

  <details>
    <summary><strong style="font-size: 1.5em;">📄 <a href="legacy/hkely-rag-system-chat.py">hkely-rag-system-chat.py</a></strong></summary>
    <div style="margin-left: 20px;">
    Το παρόν script δημιουργεί ένα έξυπνο σύστημα ερωταποκρίσεων (RAG) για την πλατφόρμα ΗΚΕΛΥ. Φορτώνει το εγχειρίδιο χρήστη (PDF) και ένα αρχείο δεδομένων (JSON) με ερωταπαντήσεις και μεταδεδομένα, εκπαιδεύει ένα σύστημα αναζήτησης και συνδυάζει τις πληροφορίες με το meltemi που έχει γίνει finetuned. Ο χρήστης μπορεί να υποβάλει ερωτήσεις (σε ελληνικά) και το σύστημα απαντά χρησιμοποιώντας τα πιο σχετικά αποσπάσματα από το εγχειρίδιο και τα δεδομένα, παρουσιάζοντας ταυτόχρονα τις πηγές της απάντησης
    </div>
  </details>
  </div>
</details>