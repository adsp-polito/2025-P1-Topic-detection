import wandb

from config import cfg


class WandBLogger:
    """
    Wrapper for WandB logging to centralize logic and configurations.
    """

    def __init__(self, job_type: str = "pipeline", run_name: str = None):
        self.enabled = cfg.get("project.wandb_logging", False)
        self.run = None

        if self.enabled:
            # Is a run already active?
            if wandb.run is not None:
                self.run = wandb.run
            else:
                # No run active, start the "Main" run
                self.run = wandb.init(
                    project=cfg.get("project.name"),
                    job_type=job_type,
                    name=run_name or f"{job_type}_run",
                    config=cfg.data,
                )

    def log_metrics(self, metrics: dict):
        """Logs a dictionary of scalar metrics."""
        if self.enabled and self.run:
            self.run.log(metrics)

    def log_plot(self, plot_name: str, plot_obj, plot_type: str = "html"):
        """
        Logs plots.
        plot_type: 'html' (for interactive plotly/bokeh) or 'image' (for matplotlib)
        """
        if self.enabled and self.run:
            if plot_type == "html":
                # Convert Plotly fig to HTML if it isn't already a string
                html_content = (
                    plot_obj if isinstance(plot_obj, str) else plot_obj.to_html()
                )
                self.run.log({plot_name: wandb.Html(html_content)})
            elif plot_type == "table":
                self.run.log({plot_name: plot_obj})
            elif plot_type == "chart":
                self.run.log({plot_name: plot_obj})

    def finish(self):
        """Closes the run."""
        if self.enabled and self.run:
            self.run.finish()
