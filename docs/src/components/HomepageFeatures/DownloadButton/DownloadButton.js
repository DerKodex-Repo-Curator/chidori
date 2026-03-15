import React, { useState, useEffect } from "react";
import { Avatar } from '@mui/material';
import styles from "./DownloadButton.module.css";
import Link from "@docusaurus/Link";
import { useHistory } from "@docusaurus/router";
import { Button } from "@mui/material";

function OSButton({ props }) {
  const history = useHistory();
  return (
    <div>
      <Link to={props.href} className={styles.link}
        onClick={e => { if (props.disabled) { e.preventDefault(); } }}
        style={{ cursor: props.disabled ? 'default' : 'auto' }} aria-disabled={props.disabled}
      >
        <Button
          variant="contained"
          className={styles.customBtn}
          disabled={props.disabled}
          sx={{
            width: "310px",
            background: " #410099", '&:hover': { backgroundColor: '#5400c7' }, '&:disabled': { backgroundColor: '#410099 !important' },
            color: "#ffffff !important",
            borderStyle: "none",
            display: "flex",
            fontFamily: '"Noto Sans", "Roboto", "Helvetica", "Arial", sans-serif'
            // Adding Noto Sans to the default MUI font list works around an alignment bug that affects Firefox on Linux.
            // See:
            // - https://github.com/primer/css/issues/1209
            // - https://bugs.launchpad.net/ubuntu/+source/firefox/+bug/1874842
          }}
          onClick={() => {
            history.push(props.url);
            dataLayer.push({
              event: "e_document",
              documentAction: "download",
              documentID: props.documentID,
              documentType: props.documentType,
            });
          }}

        >
          <div
            style={{
              display: "flex",
              justifyContent: "flex-start",
              alignItems: "center",
              minHeight: "34px",
            }}
          >
            <div>
              <Avatar className={styles.downloadIcon} sx={{ fontSize: "50px" }} variant='square' src={props.logo} alt={props.logoTitle} />
            </div>
            <span style={{ margin: 0, fontSize: '16px', fontWeight: 'bold' }}> {props.disabled || "Download for"} {props.title}</span>
          </div>
        </Button>
      </Link>
    </div>
  );
}

export default function DownloadButton(props) {
  const [os, setOs] = useState("");

  useEffect(() => {
    const userAgent = window.navigator.userAgent;
    const platform = userAgent.match(/Windows/i) ? "Windows" : "Linux";
    setOs(platform);
  }, []);

  return (
    <div>
      {props.list.map((ele) => (
        ele.os === os ? <OSButton key={ele.title} props={ele} /> : null)
      )}
    </div>
  );
}
