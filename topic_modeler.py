import re

import nltk
import torch
import transformers
import wandb
from bertopic import BERTopic
from bertopic.representation import TextGeneration
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import KernelPCA
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from umap import UMAP

from config import cfg


class TopicModeler:
    """
    BERTopic wrapper with SOTA Italian configurations and WandB logging.
    """

    ARCHITECTURES = {
        # Default
        "umap_hdbscan": {"reduction": "umap", "clustering": "hdbscan"},
        "kernelpca_spectral": {"reduction": "kernel_pca", "clustering": "spectral"},
        "kernelpca_kmeans": {"reduction": "kernel_pca", "clustering": "kmeans"},
        "umap_spectral": {"reduction": "umap", "clustering": "spectral"},
        "umap_kmeans": {"reduction": "umap", "clustering": "kmeans"},
    }

    RUN_TYPES = {
        "unsupervised": {
            "use_labels": False,
            "use_seed_topics": False,
        },
        "guided": {
            "use_labels": False,
            "use_seed_topics": True,
        },
        "semi_supervised": {
            "use_labels": True,
            "use_seed_topics": False,
        },
    }

    def __init__(self, umap_params=None, hdbscan_params=None):
        self.config = cfg.get("topic_modeling")
        self.project_name = cfg.get("project.name")
        self.embedding_model = None
        self.topic_model = None

        # Hyperparameter tuning (only used if avaiable)
        self.umap_params = umap_params
        self.hdbscan_params = hdbscan_params

        # Setup Italian Stopwords
        nltk.download("stopwords", quiet=True)
        self.stop_words = stopwords.words("italian")
        # Add project-specific noise words
        extras = self.config.get("extra_stopwords", [])
        self.stop_words.extend(extras)

    def get_reduction(self, name):
        if name == "umap":
            if self.umap_params is not None:
                ucfg = self.umap_params
            else:
                ucfg = self.config.get("umap")

            return UMAP(
                n_neighbors=ucfg["n_neighbors"],
                n_components=ucfg["n_components"],
                min_dist=ucfg["min_dist"],
                metric="cosine",
                random_state=cfg.get("project.seed", 42),
            )

        if name == "kernel_pca":
            kcfg = self.config.get("kernel_pca")
            return KernelPCA(
                n_components=kcfg["n_components"],
                kernel=kcfg.get("kernel", "rbf"),
                random_state=cfg.get("project.seed", 42),
            )

        raise ValueError(f"Unknown reduction: {name}")

    def get_clustering(self, name):
        if name == "hdbscan":
            if self.hdbscan_params is not None:
                hcfg = self.hdbscan_params
            else:
                hcfg = self.config.get("hdbscan")

            return HDBSCAN(
                min_cluster_size=hcfg["min_cluster_size"],
                min_samples=hcfg.get("min_samples", None),
                metric="euclidean",
                cluster_selection_method=hcfg.get("cluster_selection_method", "eom"),
                prediction_data=True,
            )

        if name == "kmeans":
            kcfg = self.config.get("kmeans")
            return KMeans(
                n_clusters=kcfg["n_clusters"],
                random_state=cfg.get("project.seed", 42),
                n_init="auto",
            )

        if name == "spectral":
            scfg = self.config.get("spectral")
            return SpectralClustering(
                n_clusters=scfg["n_clusters"],
                affinity="nearest_neighbors",
                random_state=cfg.get("project.seed", 42),
            )

        raise ValueError(f"Unknown clustering: {name}")

    def clean_label(self, raw_label: str) -> str:
        label = " ".join(raw_label.split())

        unwanted_prefixes = [
            r"^(Risposta|Output|Label|Etichetta|Categoria):\s*",
            r'^["\']',
        ]
        for pattern in unwanted_prefixes:
            label = re.sub(pattern, "", label, flags=re.IGNORECASE)

        match = re.match(
            r"(Problemi\s+(?:di|con|per)\s+[\w\s]{1,30}?)(?:\s+[A-Z]|\.|,|\s+La\b|\s+Il\b|\s+D\b|\s+Dopo\b)",
            label,
            re.IGNORECASE,
        )

        if match:
            label = match.group(1).strip()
        else:
            words = label.split()
            if words[0].lower() == "problemi":
                label = " ".join(words[:4])

        label = re.sub(r'[.,;:!?\'"]+$', "", label)

        label = label[0].upper() + label[1:].lower() if label else label

        if not label.lower().startswith("problemi"):
            label = f"Problemi di {label.lower()}"

        return label

    def run(
        self,
        docs: list,
        y=None,
        run_name: str = "bertopic_run",
        architecture_name: str = "umap_hdbscan",
        type_name: str = "unsupervised",
        logger=None,
    ):
        """
        Executes the topic modeling pipeline.
        """
        model_type = self.config.get("model_choice")
        print(f"--> [BERTopic] Starting run: {run_name} using [{model_type}] config")

        architecture = self.ARCHITECTURES[architecture_name]
        print(f"    Running architecture: {architecture}")

        run_cfg = self.RUN_TYPES[type_name]
        print(f"    Running type: {type_name}")

        # Wandb
        if cfg.get("project.wandb_logging") and logger is None:
            print("    Warning: WandBLogger must be passed from main()")

        # 1. Select and Store Embedding Model
        # We store it in 'self' to access it later for evaluation
        model_name = self.config.get("embedding_model")
        print(f"    Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)

        # Encode embeddings
        print("    Encoding embeddings...")
        embeddings = self.embedding_model.encode(docs, show_progress_bar=True)

        # 2. Dimensionality Reduction
        print(
            f"    Loading dimensionality reduction model ({architecture['reduction']})..."
        )
        dim_red_model = self.get_reduction(architecture["reduction"])

        # 3. Clustering
        print(f"    Loading clustering model ({architecture['clustering']})...")
        cluster_model = self.get_clustering(architecture["clustering"])

        # 4. Vectorizer (Topic Representation) min_df=2 ensures words appear in at least 2 topics (or docs), preventing the crash if topics are few.
        vectorizer_model = CountVectorizer(
            stop_words=self.stop_words, min_df=self.config.get("min_df", 2)
        )

        # 5. Definition of the type of running
        if type_name not in self.RUN_TYPES:
            raise ValueError(f"Unknown run type: {type_name}")

        run_cfg = self.RUN_TYPES[type_name]
        print(f"    Running type: {type_name}")

        seed_topic_list = None
        if run_cfg["use_seed_topics"]:
            seed_topic_list = self.config.get("seed_topics", [])
            print("    First 3 seed topics:")
            for i, seed in enumerate(seed_topic_list[:3]):
                print(f"      Topic {i}: {seed[:5]}...")

        # 6. Initialize BERTopic (ONCE)
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=dim_red_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            seed_topic_list=seed_topic_list,
            language=self.config.get("language", "multilingual"),
            calculate_probabilities=architecture_name == "umap_hdbscan",
            verbose=True,
            nr_topics=15,
        )

        print("--> [BERTopic] Fitting model...")

        # 7. Fit model (semi-supervised if required)
        if run_cfg["use_labels"]:
            if y is None:
                raise ValueError(
                    "Semi-supervised mode requires y labels, but y=None was passed"
                )

            if len(y) != len(docs):
                raise ValueError(
                    f"y must have same length as docs. Got y={len(y)}, docs={len(docs)}"
                )

            if architecture_name == "umap_hdbscan":
                topics, probs = self.topic_model.fit_transform(
                    docs, embeddings=embeddings, y=y
                )
            else:
                topics = self.topic_model.fit_transform(
                    docs, embeddings=embeddings, y=y
                )
                probs = None
        else:
            if architecture_name == "umap_hdbscan":
                topics, probs = self.topic_model.fit_transform(
                    docs, embeddings=embeddings
                )
            else:
                topics = self.topic_model.fit_transform(docs, embeddings=embeddings)
                probs = None

        if cfg.get("reductionNumberTopic"):
            # 8. Merge topics
            fig = self.topic_model.visualize_topics()
            fig.write_html("./out/topic.html")

            # Merge by number of topics
            numberOfTopics = input(
                "Look at topic.html, If you want to merge topics insert a number (+1), 0 otherwise"
            )
            if numberOfTopics.isdigit():
                nr = int(numberOfTopics)
                if nr != 0:
                    self.topic_model.reduce_topics(docs, nr_topics=nr)
                    topics = self.topic_model.topics_

                    if architecture_name == "umap_hdbscan":
                        _, probs = self.topic_model.transform(docs)
                    else:
                        _ = self.topic_model.transform(docs)
                        probs = None

            newfig = self.topic_model.visualize_topics()
            newfig.write_html("./out/newTopic.html")

            # Merge by specific topics
            merge = input(
                "Look at topic.html. Do you want to merge specific topics? (Y/N): "
            )

            while merge.upper() == "Y":
                print(
                    "Insert topic pairs to merge, one per line.\n"
                    "Example:\n"
                    "1 2\n"
                    "3 4\n"
                    "Empty line to finish."
                )

                list_to_merge = []

                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    try:
                        a, b = map(int, line.split())
                        list_to_merge.append([a, b])
                    except ValueError:
                        print("Invalid format. Use: <int> <int>")

                if not list_to_merge:
                    print("No valid topic pairs provided. Skipping merge.")
                else:
                    self.topic_model.merge_topics(docs, list_to_merge)
                    topics = self.topic_model.topics_
                    if architecture_name == "umap_hdbscan":
                        _, probs = self.topic_model.transform(docs)
                    else:
                        _ = self.topic_model.transform(docs)
                        probs = None

                    newfig = self.topic_model.visualize_topics()
                    newfig.write_html("./out/newTopic.html")

                    print(f"Merged topic pairs: {list_to_merge}")

                merge = input("Look at newTopic.html. Merge more topics? (Y/N): ")

        if cfg.get("llmRepresentation"):
            print("    Loading Llama 3.1 8B Instruct...")
            model_id = "meta-llama/Llama-3.1-8B-Instruct"

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token  # Fix per padding

            # Quantizzazione
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

            prompt = """
                Sei un analista di feedback utente esperto. Il tuo compito è sintetizzare recensioni e parole chiave in un'unica categoria standardizzata.

                INPUT DA ANALIZZARE:
                REVIEWS: [DOCUMENTS]
                KEYWORDS: [KEYWORDS]

                REGOLE TASSATIVE:
                1. Produci UNICAMENTE una label in italiano di massimo 4 parole.
                2. Inizia sempre con la formula "Problemi di" (o "Problemi con" se più naturale).
                3. Priorità: Se l'input contiene log tecnici (GPU, pipeline, codice) e recensioni utente, IGNORE i log tecnici e categorizza solo il disagio dell'utente.
                4. Formattazione: Solo testo, niente punteggiatura finale, niente grassetti, niente virgolette, niente prefissi come "Label:" o "Risposta:".
                5. Se i documenti contengono più problemi, identifica quello principale o più frequente.
                6. Non inserire nella tua risposta errori tipo "Spiega", "Sp", "Etichetta" o "Giusto", basati sulla formula.

                ESEMPI:
                Input: "L'app crasha sempre al login. Errore GPU pipeline 0x01" -> Problemi di accesso app
                Input: "Non riesco a pagare con la ricaricabile" -> Problemi di pagamenti
                Input: "Lento a caricare le notifiche" -> Problemi di notifiche

                OUTPUT:
            """

            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=8,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
                return_full_text=False,
            )
            topic_labels = {}
            for topic_id in range(len(self.topic_model.get_topic_info()) - 1):
                topic_words = [
                    word for word, _ in self.topic_model.get_topic(topic_id)[:10]
                ]
                topic_docs = [
                    doc for doc, topic in zip(docs, topics) if topic == topic_id
                ][:5]

                prompt_filled = prompt.replace("[DOCUMENTS]", "\n".join(topic_docs[:3]))
                prompt_filled = prompt_filled.replace(
                    "[KEYWORDS]", ", ".join(topic_words)
                )

                result = generator(prompt_filled)
                label = result[0]["generated_text"].strip()
                clean_label_text = self.clean_label(label)
                print(f"Topic {topic_id}: {label} -> {clean_label_text}")
                topic_labels[topic_id] = clean_label_text

            self.topic_model.set_topic_labels(topic_labels)

            representation_model = TextGeneration(generator, prompt=prompt)

            self.topic_model.update_topics(
                docs, representation_model=representation_model
            )

        fig_hierarchy = self.topic_model.visualize_hierarchy(
            top_n_topics=50, custom_labels=True
        )
        fig_hierarchy.write_html("./out/fig_hierarchy.html")

        # 9. Logging to WandB
        freq = self.topic_model.get_topic_info()
        n_topics = len(freq) - 1
        print(f"--> [BERTopic] Generated {n_topics} topics.")

        if logger is not None:
            logger.log_metrics({"n_topics": n_topics, "n_docs": len(docs)})
            # Log Topic Info Table
            wandb.log({"topic_info": wandb.Table(dataframe=freq)})

            # --- ADVANCED PLOTS ---
            try:
                # A. Intertopic Distance
                logger.log_plot("plot_intertopic", self.topic_model.visualize_topics())

                # B. Bar Chart (Top 15)
                logger.log_plot(
                    "plot_barchart",
                    self.topic_model.visualize_barchart(top_n_topics=15),
                )

                # C. Hierarchy (Tree) - Addresses Task 3b
                # Limit to top 50 topics to keep it readable
                fig_hierarchy = self.topic_model.visualize_hierarchy(top_n_topics=50)
                logger.log_plot("plot_hierarchy", fig_hierarchy)

                # D. Similarity Heatmap - Addresses Topic Separation
                # We filter out Topic -1 (Noise) for the heatmap
                if n_topics > 1:
                    fig_heatmap = self.topic_model.visualize_heatmap(
                        n_clusters=n_topics - 1
                    )
                    logger.log_plot("plot_heatmap", fig_heatmap)

            except Exception as e:
                print(f"--> [Warning] Could not log plots: {e}")

        return self.topic_model, topics, probs

    def save_model(self, path: str):
        """Saves the trained model to disk."""
        if self.topic_model:
            # Safetensors is preferred for security and speed
            self.topic_model.save(path, serialization="safetensors", save_ctfidf=True)
            print(f"--> [BERTopic] Model saved locally to {path}")
        else:
            print("--> [Error] No model to save. Run .run() first.")
